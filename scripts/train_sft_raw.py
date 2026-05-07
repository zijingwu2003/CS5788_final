"""SFT training — raw PyTorch loop, no TRL/Trainer.

Loss: teacher-forcing cross-entropy on response tokens only.
The prompt tokens are masked out so the model is only trained on chosen replies.

Usage (smoke-test):
    python scripts/train_sft_raw.py --max-samples 200 --epochs 1 --fp16

Full run:
    python scripts/train_sft_raw.py --fp16
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


class SFTDataset(Dataset):
    """Tokenises prompt+chosen pairs; labels mask out prompt tokens."""

    def __init__(
        self,
        rows: list[dict],
        tokenizer: AutoTokenizer,
        max_length: int,
        max_prompt_length: int,
    ) -> None:
        self.samples = []
        for row in rows:
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=True)
            if len(prompt_ids) > max_prompt_length:
                prompt_ids = prompt_ids[:max_prompt_length]

            remaining = max_length - len(prompt_ids)
            response_ids = tokenizer.encode(
                " " + row["chosen"], add_special_tokens=False
            )[:remaining]

            if not response_ids:
                continue

            input_ids = prompt_ids + response_ids
            # -100 masks prompt tokens; only response tokens contribute to loss
            labels = [-100] * len(prompt_ids) + response_ids
            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class SFTCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        def pad(seqs: list[list[int]], pad_val: int) -> torch.Tensor:
            max_len = max(len(s) for s in seqs)
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs])

        input_ids = pad([x["input_ids"] for x in batch], self.pad_id)
        labels    = pad([x["labels"]    for x in batch], -100)
        attn_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Teacher-forcing cross-entropy on non-masked label positions.

    logits : [B, T, V]
    labels : [B, T]  (-100 = ignore)
    """
    shift_logits = logits[:, :-1, :]        # predict token t+1 from token t
    shift_labels = labels[:, 1:].clone()    # [B, T-1]
    mask = shift_labels != -100             # [B, T-1]

    shift_labels[~mask] = 0                 # safe index for cross_entropy
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    )
    mask_flat = mask.reshape(-1).float()
    return (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)


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
    parser = argparse.ArgumentParser(description="Raw SFT training (no TRL)")
    parser.add_argument("--model-name",        default="gpt2")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_train.jsonl"))
    parser.add_argument("--eval-file",  type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/sft"))
    parser.add_argument("--epochs",            type=int,   default=1)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--grad-accum",        type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=2e-5)
    parser.add_argument("--max-length",        type=int,   default=512)
    parser.add_argument("--max-prompt-length", type=int,   default=256)
    parser.add_argument("--max-samples",       type=int,   default=None)
    parser.add_argument("--log-steps",         type=int,   default=50)
    parser.add_argument("--fp16",              action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    train_ds = SFTDataset(train_rows, tokenizer, args.max_length, args.max_prompt_length)
    eval_ds  = SFTDataset(eval_rows,  tokenizer, args.max_length, args.max_prompt_length)
    collator = SFTCollator(tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collator)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    print(f"Train: {len(train_ds)} examples | Eval: {len(eval_ds)} examples")

    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: cosine_schedule(s, total_steps))
    scaler    = torch.cuda.amp.GradScaler(enabled=use_fp16)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                out  = model(input_ids=input_ids, attention_mask=attn_mask)
                loss = compute_sft_loss(out.logits, labels) / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_steps == 0:
                    print(f"epoch {epoch} | step {global_step}/{total_steps} | loss {running_loss / args.log_steps:.4f}")
                    running_loss = 0.0

        # Eval
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["labels"].to(device)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    out  = model(input_ids=input_ids, attention_mask=attn_mask)
                    loss = compute_sft_loss(out.logits, labels)
                eval_losses.append(loss.item())
        avg_eval = sum(eval_losses) / len(eval_losses)
        print(f"epoch {epoch} | eval_loss {avg_eval:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"SFT model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
