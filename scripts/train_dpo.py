"""DPO training: start from the SFT checkpoint and apply Direct Preference Optimization.

DPO optimises the log-likelihood ratio between the trainable policy π_θ and a
frozen reference model π_ref (both initialised from the SFT checkpoint).

Usage (smoke-test):
    python scripts/train_dpo.py --max-samples 200 --epochs 1 --fp16

Full run:
    python scripts/train_dpo.py --fp16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training starting from the SFT checkpoint")
    parser.add_argument(
        "--model-name", default="models/sft",
        help="SFT checkpoint path or HuggingFace model name used as the trainable policy",
    )
    parser.add_argument(
        "--ref-model-name", default=None,
        help="Frozen reference model (defaults to --model-name, i.e. the same SFT checkpoint)",
    )
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_train.jsonl"))
    parser.add_argument("--eval-file",  type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/dpo"))
    parser.add_argument("--epochs",           type=int,   default=1)
    # DPO holds two model copies in memory; use smaller batch + more grad-accum
    parser.add_argument("--batch-size",       type=int,   default=2)
    parser.add_argument("--grad-accum",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=1e-5)
    parser.add_argument("--beta",             type=float, default=0.1, help="KL-penalty coefficient β")
    parser.add_argument("--max-length",       type=int,   default=512, help="Max total sequence length (prompt+response)")
    parser.add_argument("--max-prompt-length", type=int,  default=256, help="Max prompt length")
    parser.add_argument("--max-samples",      type=int,   default=None, help="Cap dataset size (debugging only)")
    parser.add_argument("--fp16", action="store_true", help="Mixed-precision training (GPU only)")
    args = parser.parse_args()

    ref_model_name = args.ref_model_name or args.model_name

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model     = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)

    train_rows = load_jsonl(args.train_file)
    eval_rows  = load_jsonl(args.eval_file)

    if args.max_samples is not None:
        train_rows = train_rows[: args.max_samples]
        eval_rows  = eval_rows[: max(1, args.max_samples // 10)]

    # DPOTrainer expects columns: prompt, chosen, rejected
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset  = Dataset.from_list(eval_rows)

    print(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

    config = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=args.fp16,
        bf16=False,
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"DPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
