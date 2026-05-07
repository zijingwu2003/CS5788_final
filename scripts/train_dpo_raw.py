"""DPO training — raw PyTorch loop, no TRL/Trainer.

Starts from the SFT checkpoint and trains with Direct Preference Optimization.

Loss:  -log σ(β · [(log π_θ(c|x) − log π_θ(r|x)) − (log π_ref(c|x) − log π_ref(r|x))])

where log-prob is the *sum* over response (non-prompt) tokens.

Usage (smoke-test):
    python scripts/train_dpo_raw.py --max-samples 200 --epochs 1 --fp16

Full run:
    python scripts/train_dpo_raw.py --fp16
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class DPODataset(Dataset):
    """Stores raw strings; tokenisation is deferred to the collator."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class DPOCollator:
    """Tokenise chosen/rejected full sequences and record prompt length."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        max_prompt_length: int,
        device: torch.device,
    ) -> None:
        self.tokenizer        = tokenizer
        self.max_length       = max_length
        self.max_prompt_length = max_prompt_length
        self.device           = device

    def _encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def __call__(self, batch: list[dict]) -> dict:
        # Concatenate prompt + response for each side
        chosen_texts   = [b["prompt"] + " " + b["chosen"]   for b in batch]
        rejected_texts = [b["prompt"] + " " + b["rejected"] for b in batch]

        chosen   = self._encode(chosen_texts)
        rejected = self._encode(rejected_texts)

        # Prompt-only length (used to build the completion mask)
        prompt_only = [b["prompt"] for b in batch]
        prompt_enc  = self.tokenizer(
            prompt_only,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        prompt_lens = torch.tensor(
            [min(len(ids), self.max_prompt_length) for ids in prompt_enc["input_ids"]],
            dtype=torch.long,
            device=self.device,
        )

        return {
            "chosen_input_ids":        chosen["input_ids"],
            "chosen_attention_mask":   chosen["attention_mask"],
            "rejected_input_ids":      rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "prompt_lens":             prompt_lens,
        }


# ---------------------------------------------------------------------------
# Log-probability helpers
# ---------------------------------------------------------------------------

def _completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Binary mask [B, T-1]: 1 for response tokens, 0 for prompt/padding.

    We score positions t where:
      - t >= prompt_len  (response, not prompt)
      - attention_mask[:, t+1] == 1  (not padding in the target)
    We use the shifted view: logits[:, :-1] predicts input_ids[:, 1:],
    so the mask also has length T-1.
    """
    B, T = input_ids.shape
    pos  = torch.arange(T - 1, device=input_ids.device)           # [T-1]
    # position t in the logit dimension corresponds to predicting token t+1
    after_prompt = pos.unsqueeze(0) >= prompt_lens.unsqueeze(1)    # [B, T-1]
    not_pad      = attention_mask[:, 1:].bool()                    # [B, T-1]
    return (after_prompt & not_pad).float()


def sequence_logprob(
    model_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Sum of log-probs over completion tokens.

    model_logits : [B, T, V]
    Returns      : [B]
    """
    log_probs = F.log_softmax(model_logits[:, :-1, :], dim=-1)   # [B, T-1, V]
    targets   = input_ids[:, 1:].clone()                          # [B, T-1]
    mask      = _completion_mask(input_ids, attention_mask, prompt_lens)  # [B, T-1]

    # Gather log-prob of the actual next token
    selected = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    selected = selected * mask
    return selected.sum(-1)                                        # [B]


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def compute_dpo_loss(
    chosen_logprob:     torch.Tensor,
    rejected_logprob:   torch.Tensor,
    chosen_logprob_ref: torch.Tensor,
    rejected_logprob_ref: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Batch-mean DPO loss.

    All inputs are [B] tensors of total completion log-probabilities.
    """
    logits = beta * (
        (chosen_logprob   - rejected_logprob) -
        (chosen_logprob_ref - rejected_logprob_ref)
    )
    return -F.logsigmoid(logits).mean()


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def cosine_schedule(step: int, total: int, min_ratio: float = 0.1) -> float:
    if total <= 1:
        return 1.0
    p = step / max(1, total - 1)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Raw DPO training (no TRL)")
    parser.add_argument("--model-name",     default="models/sft",
                        help="SFT checkpoint used as the trainable policy")
    parser.add_argument("--ref-model-name", default=None,
                        help="Frozen reference (defaults to --model-name)")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_train.jsonl"))
    parser.add_argument("--eval-file",  type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/dpo"))
    parser.add_argument("--epochs",            type=int,   default=1)
    parser.add_argument("--batch-size",        type=int,   default=2)
    parser.add_argument("--grad-accum",        type=int,   default=8)
    parser.add_argument("--lr",                type=float, default=1e-5)
    parser.add_argument("--beta",              type=float, default=0.1)
    parser.add_argument("--max-length",        type=int,   default=512)
    parser.add_argument("--max-prompt-length", type=int,   default=256)
    parser.add_argument("--max-samples",       type=int,   default=None)
    parser.add_argument("--log-steps",         type=int,   default=50)
    parser.add_argument("--fp16",              action="store_true")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16  = args.fp16 and torch.cuda.is_available()
    ref_name  = args.ref_model_name or args.model_name

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy    = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    train_rows = load_jsonl(args.train_file)
    eval_rows  = load_jsonl(args.eval_file)
    if args.max_samples is not None:
        train_rows = train_rows[: args.max_samples]
        eval_rows  = eval_rows[: max(1, args.max_samples // 10)]

    collator = DPOCollator(tokenizer, args.max_length, args.max_prompt_length, device)

    train_loader = DataLoader(
        DPODataset(train_rows), batch_size=args.batch_size,
        shuffle=True, collate_fn=collator,
    )
    eval_loader = DataLoader(
        DPODataset(eval_rows), batch_size=args.batch_size,
        shuffle=False, collate_fn=collator,
    )

    print(f"Train: {len(train_rows)} examples | Eval: {len(eval_rows)} examples")

    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    optimizer   = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler   = LambdaLR(optimizer, lr_lambda=lambda s: cosine_schedule(s, total_steps))
    scaler      = torch.cuda.amp.GradScaler(enabled=use_fp16)

    global_step   = 0
    running_loss  = 0.0
    running_margin = 0.0

    for epoch in range(args.epochs):
        policy.train()
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            c_ids  = batch["chosen_input_ids"]
            c_attn = batch["chosen_attention_mask"]
            r_ids  = batch["rejected_input_ids"]
            r_attn = batch["rejected_attention_mask"]
            plens  = batch["prompt_lens"]

            # Reference log-probs (no gradient)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
                c_logits_ref = ref_model(input_ids=c_ids, attention_mask=c_attn).logits
                r_logits_ref = ref_model(input_ids=r_ids, attention_mask=r_attn).logits
            c_lp_ref = sequence_logprob(c_logits_ref, c_ids, c_attn, plens)
            r_lp_ref = sequence_logprob(r_logits_ref, r_ids, r_attn, plens)

            # Policy log-probs (with gradient)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                c_logits_pi = policy(input_ids=c_ids, attention_mask=c_attn).logits
                r_logits_pi = policy(input_ids=r_ids, attention_mask=r_attn).logits

            c_lp_pi = sequence_logprob(c_logits_pi, c_ids, c_attn, plens)
            r_lp_pi = sequence_logprob(r_logits_pi, r_ids, r_attn, plens)

            loss = compute_dpo_loss(c_lp_pi, r_lp_pi, c_lp_ref, r_lp_ref, args.beta)
            scaler.scale(loss / args.grad_accum).backward()

            with torch.no_grad():
                margin = args.beta * ((c_lp_pi - c_lp_ref) - (r_lp_pi - r_lp_ref))
                running_loss   += loss.item()
                running_margin += margin.mean().item()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_steps == 0:
                    denom = args.log_steps * args.grad_accum
                    print(
                        f"epoch {epoch} | step {global_step}/{total_steps} "
                        f"| loss {running_loss / denom:.4f} "
                        f"| reward_margin {running_margin / denom:.4f}"
                    )
                    running_loss = running_margin = 0.0

        # Eval
        policy.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                c_ids  = batch["chosen_input_ids"]
                c_attn = batch["chosen_attention_mask"]
                r_ids  = batch["rejected_input_ids"]
                r_attn = batch["rejected_attention_mask"]
                plens  = batch["prompt_lens"]

                with torch.cuda.amp.autocast(enabled=use_fp16):
                    c_lp_ref = sequence_logprob(
                        ref_model(input_ids=c_ids, attention_mask=c_attn).logits,
                        c_ids, c_attn, plens,
                    )
                    r_lp_ref = sequence_logprob(
                        ref_model(input_ids=r_ids, attention_mask=r_attn).logits,
                        r_ids, r_attn, plens,
                    )
                    c_lp_pi = sequence_logprob(
                        policy(input_ids=c_ids, attention_mask=c_attn).logits,
                        c_ids, c_attn, plens,
                    )
                    r_lp_pi = sequence_logprob(
                        policy(input_ids=r_ids, attention_mask=r_attn).logits,
                        r_ids, r_attn, plens,
                    )
                    eloss = compute_dpo_loss(c_lp_pi, r_lp_pi, c_lp_ref, r_lp_ref, args.beta)
                eval_losses.append(eloss.item())

        avg_eval = sum(eval_losses) / len(eval_losses)
        print(f"epoch {epoch} | eval_loss {avg_eval:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"DPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
