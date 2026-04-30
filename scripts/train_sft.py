"""SFT baseline: fine-tune a decoder-only model on HH-RLHF chosen responses.

Usage (small smoke-test):
    python scripts/train_sft.py --max-samples 200 --epochs 1 --fp16

Full run (default GPT-2 Small):
    python scripts/train_sft.py --fp16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT baseline on HH-RLHF chosen responses")
    parser.add_argument(
        "--model-name", default="gpt2",
        help="HuggingFace model name or local path (e.g. gpt2, gpt2-medium, EleutherAI/pythia-410m)",
    )
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_train.jsonl"))
    parser.add_argument("--eval-file",  type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/sft"))
    parser.add_argument("--epochs",     type=int,   default=1)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--grad-accum", type=int,   default=4,   help="Gradient accumulation steps")
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max-length", type=int,   default=512, help="Max tokens per training example")
    parser.add_argument("--max-samples", type=int,  default=None, help="Cap dataset size (debugging only)")
    parser.add_argument("--fp16", action="store_true", help="Mixed-precision training (GPU only)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    train_rows = load_jsonl(args.train_file)
    eval_rows  = load_jsonl(args.eval_file)

    if args.max_samples is not None:
        train_rows = train_rows[: args.max_samples]
        eval_rows  = eval_rows[: max(1, args.max_samples // 10)]

    # Concatenate prompt + chosen into a single training sequence.
    # The prompt already ends with "Assistant:" so the full text is natural.
    train_dataset = Dataset.from_list([{"text": r["prompt"] + " " + r["chosen"]} for r in train_rows])
    eval_dataset  = Dataset.from_list([{"text": r["prompt"] + " " + r["chosen"]} for r in eval_rows])

    print(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

    config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        dataset_text_field="text",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=100,
        fp16=args.fp16,
        bf16=False,
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"SFT model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
