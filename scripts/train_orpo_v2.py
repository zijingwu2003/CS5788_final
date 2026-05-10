"""ORPO v2 — raw PyTorch loop, no TRL/Trainer.

Improvements over train_orpo.py:
  1. Warm-starts from SFT checkpoint (default --model-name models/sft)
  2. Tokenises prompt+response as a single string (BPE-friendly, consistent
     with train_sft_raw / train_dpo_raw)
  3. Higher default β=0.2 for stronger preference signal
  4. Logs accuracy (fraction of pairs where log_OR > 0) and OR margin

Loss:  L = L_SFT + β · L_OR
  L_SFT = cross-entropy on response tokens only (teacher forcing)
  L_OR  = −log σ(log_odds(chosen) − log_odds(rejected))
  log_odds(y|x) = log[P(y|x) / (1 − P(y|x))]
  where P is the mean per-token probability over response tokens.

Usage (smoke-test):
    python scripts/train_orpo_v2.py --max-samples 200 --epochs 1 --fp16

Full run (from SFT checkpoint):
    python scripts/train_orpo_v2.py --fp16

Full run from base GPT-2 (no SFT):
    python scripts/train_orpo_v2.py --model-name gpt2 --fp16
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


class ORPODataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


class ORPOCollator:
    """Tokenise chosen and rejected as single strings; record prompt length."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        max_prompt_length: int,
        device: torch.device,
    ) -> None:
        self.tokenizer         = tokenizer
        self.max_length        = max_length
        self.max_prompt_length = max_prompt_length
        self.device            = device

    def _encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def __call__(self, batch: list[dict]) -> dict:
        chosen_texts   = [b["prompt"] + " " + b["chosen"]   for b in batch]
        rejected_texts = [b["prompt"] + " " + b["rejected"] for b in batch]

        chosen   = self._encode(chosen_texts)
        rejected = self._encode(rejected_texts)

        prompt_enc = self.tokenizer(
            [b["prompt"] for b in batch],
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
# Masking helper
# ---------------------------------------------------------------------------

def response_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """[B, T-1] float mask: 1 for response tokens in the shifted view."""
    T   = input_ids.shape[1]
    pos = torch.arange(T - 1, device=input_ids.device)        # [T-1]
    after_prompt = pos.unsqueeze(0) >= prompt_lens.unsqueeze(1)  # [B, T-1]
    not_pad      = attention_mask[:, 1:].bool()                  # [B, T-1]
    return (after_prompt & not_pad).float()


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def compute_sft_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy on chosen response tokens only."""
    shift_logits  = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:].clone()
    mask          = response_mask(input_ids, attention_mask, prompt_lens)

    shift_targets[mask == 0] = 0
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        reduction="none",
    )
    mask_flat = mask.reshape(-1)
    return (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)


def mean_log_prob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Mean per-token log-prob over response tokens. Returns [B]."""
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
    targets   = input_ids[:, 1:].clone()                   # [B, T-1]
    mask      = response_mask(input_ids, attention_mask, prompt_lens)

    targets[mask == 0] = 0
    selected  = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    selected  = selected * mask
    return selected.sum(-1) / mask.sum(-1).clamp(min=1.0)  # [B]


def _log_odds(log_p: torch.Tensor) -> torch.Tensor:
    """log[p / (1-p)] computed stably from log_p."""
    return log_p - torch.log(1.0 - torch.exp(log_p.clamp(max=-1e-7)) + 1e-7)


def compute_or_loss(
    chosen_log_prob: torch.Tensor,
    rejected_log_prob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Odds-ratio loss and the raw log_OR values (for diagnostics).

    Returns (or_loss [scalar], log_or [B])
    """
    log_or  = _log_odds(chosen_log_prob) - _log_odds(rejected_log_prob)
    or_loss = -F.logsigmoid(log_or).mean()
    return or_loss, log_or


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
    parser = argparse.ArgumentParser(description="ORPO v2 — raw training loop")
    parser.add_argument("--model-name",        default="models/sft",
                        help="Starting checkpoint. Use 'gpt2' to skip SFT init.")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_train.jsonl"))
    parser.add_argument("--eval-file",  type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/orpo_v2"))
    parser.add_argument("--epochs",            type=int,   default=1)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--grad-accum",        type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=1e-5)
    parser.add_argument("--beta",              type=float, default=0.2,
                        help="OR loss weight (higher = more preference pressure)")
    parser.add_argument("--max-length",        type=int,   default=512)
    parser.add_argument("--max-prompt-length", type=int,   default=256)
    parser.add_argument("--max-samples",       type=int,   default=None)
    parser.add_argument("--log-steps",         type=int,   default=50)
    parser.add_argument("--fp16",              action="store_true")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    train_rows = load_jsonl(args.train_file)
    eval_rows  = load_jsonl(args.eval_file)
    if args.max_samples is not None:
        train_rows = train_rows[: args.max_samples]
        eval_rows  = eval_rows[: max(1, args.max_samples // 10)]

    collator = ORPOCollator(tokenizer, args.max_length, args.max_prompt_length, device)

    train_loader = DataLoader(
        ORPODataset(train_rows), batch_size=args.batch_size,
        shuffle=True, collate_fn=collator,
    )
    eval_loader = DataLoader(
        ORPODataset(eval_rows), batch_size=args.batch_size,
        shuffle=False, collate_fn=collator,
    )

    print(f"Train: {len(train_rows)} examples | Eval: {len(eval_rows)} examples")
    print(f"Init from: {args.model_name} | β={args.beta}")

    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler   = LambdaLR(optimizer, lr_lambda=lambda s: cosine_schedule(s, total_steps))
    scaler      = torch.amp.GradScaler("cuda", enabled=use_fp16)

    global_step      = 0
    running_loss     = 0.0
    running_sft      = 0.0
    running_or       = 0.0
    running_acc      = 0.0
    running_margin   = 0.0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            c_ids  = batch["chosen_input_ids"]
            c_attn = batch["chosen_attention_mask"]
            r_ids  = batch["rejected_input_ids"]
            r_attn = batch["rejected_attention_mask"]
            plens  = batch["prompt_lens"]

            with torch.amp.autocast("cuda", enabled=use_fp16):
                c_out = model(input_ids=c_ids, attention_mask=c_attn)
                r_out = model(input_ids=r_ids, attention_mask=r_attn)

                sft_loss           = compute_sft_loss(c_out.logits, c_ids, c_attn, plens)
                chosen_lp          = mean_log_prob(c_out.logits, c_ids, c_attn, plens)
                rejected_lp        = mean_log_prob(r_out.logits, r_ids, r_attn, plens)
                or_loss, log_or    = compute_or_loss(chosen_lp, rejected_lp)

                loss = sft_loss + args.beta * or_loss

            scaler.scale(loss / args.grad_accum).backward()

            with torch.no_grad():
                running_loss   += loss.item()
                running_sft    += sft_loss.item()
                running_or     += or_loss.item()
                running_acc    += (log_or > 0).float().mean().item()
                running_margin += log_or.mean().item()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_steps == 0:
                    n = args.log_steps * args.grad_accum
                    print(
                        f"epoch {epoch} | step {global_step}/{total_steps} "
                        f"| loss {running_loss/n:.4f} "
                        f"| sft {running_sft/n:.4f} "
                        f"| or {running_or/n:.4f} "
                        f"| acc {running_acc/n:.3f} "
                        f"| margin {running_margin/n:.4f}"
                    )
                    running_loss = running_sft = running_or = running_acc = running_margin = 0.0

        # Eval
        model.eval()
        eval_losses, eval_accs = [], []
        with torch.no_grad():
            for batch in eval_loader:
                c_ids  = batch["chosen_input_ids"]
                c_attn = batch["chosen_attention_mask"]
                r_ids  = batch["rejected_input_ids"]
                r_attn = batch["rejected_attention_mask"]
                plens  = batch["prompt_lens"]

                with torch.amp.autocast("cuda", enabled=use_fp16):
                    c_out = model(input_ids=c_ids, attention_mask=c_attn)
                    r_out = model(input_ids=r_ids, attention_mask=r_attn)

                    sft_l          = compute_sft_loss(c_out.logits, c_ids, c_attn, plens)
                    chosen_lp      = mean_log_prob(c_out.logits, c_ids, c_attn, plens)
                    rejected_lp    = mean_log_prob(r_out.logits, r_ids, r_attn, plens)
                    or_l, log_or   = compute_or_loss(chosen_lp, rejected_lp)

                eval_losses.append((sft_l + args.beta * or_l).item())
                eval_accs.append((log_or > 0).float().mean().item())

        print(
            f"epoch {epoch} | eval_loss {sum(eval_losses)/len(eval_losses):.4f} "
            f"| eval_acc {sum(eval_accs)/len(eval_accs):.3f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"ORPO v2 model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
