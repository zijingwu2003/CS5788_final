"""ORPO training: monolithic preference optimisation without a reference model.

ORPOTrainer was removed in TRL 1.x, so this script provides a self-contained
implementation using a standard HuggingFace Trainer subclass.

ORPO loss (Hong et al., 2024):
    L = L_SFT + β · L_OR
    L_SFT  = cross-entropy on chosen response tokens
    L_OR   = -log σ(log_OR),  log_OR = log_odds(chosen) - log_odds(rejected)
    log_odds(y|x) = log P(y|x) - log(1 - P(y|x))

Usage (smoke-test):
    python scripts/train_orpo.py --max-samples 200 --epochs 1 --fp16

Full run:
    python scripts/train_orpo.py --fp16
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ORPOConfig(TrainingArguments):
    beta: float = field(default=0.1, metadata={"help": "Odds-ratio penalty weight (λ in the paper)"})
    max_length: int = field(default=512,  metadata={"help": "Max total sequence length (prompt + response)"})
    max_prompt_length: int = field(default=256, metadata={"help": "Max prompt length in tokens"})


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenize_row(
    example: dict,
    tokenizer: AutoTokenizer,
    max_length: int,
    max_prompt_length: int,
) -> dict:
    """Convert one {prompt, chosen, rejected} triplet into model inputs."""
    prompt          = example["prompt"]
    chosen_response = " " + example["chosen"]
    rejected_response = " " + example["rejected"]

    # Tokenise prompt (no extra special tokens — GPT-2 has none to add anyway)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[:max_prompt_length]

    prompt_len = len(prompt_ids)
    remaining  = max_length - prompt_len

    # Tokenise responses without re-adding BOS
    chosen_ids   = tokenizer.encode(chosen_response,   add_special_tokens=False)[:remaining]
    rejected_ids = tokenizer.encode(rejected_response, add_special_tokens=False)[:remaining]

    chosen_full   = prompt_ids + chosen_ids
    rejected_full = prompt_ids + rejected_ids

    # Mask prompt tokens in labels so loss is only computed on the response
    chosen_labels   = [-100] * prompt_len + chosen_ids
    rejected_labels = [-100] * prompt_len + rejected_ids

    return {
        "chosen_input_ids":        chosen_full,
        "chosen_attention_mask":   [1] * len(chosen_full),
        "chosen_labels":           chosen_labels,
        "rejected_input_ids":      rejected_full,
        "rejected_attention_mask": [1] * len(rejected_full),
        "rejected_labels":         rejected_labels,
    }


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class ORPODataCollator:
    pad_token_id: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        def pad(seqs: list[list[int]], pad_val: int) -> torch.Tensor:
            max_len = max(len(s) for s in seqs)
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs])

        return {
            "chosen_input_ids":        pad([x["chosen_input_ids"]        for x in batch], self.pad_token_id),
            "chosen_attention_mask":   pad([x["chosen_attention_mask"]   for x in batch], 0),
            "chosen_labels":           pad([x["chosen_labels"]           for x in batch], -100),
            "rejected_input_ids":      pad([x["rejected_input_ids"]      for x in batch], self.pad_token_id),
            "rejected_attention_mask": pad([x["rejected_attention_mask"] for x in batch], 0),
            "rejected_labels":         pad([x["rejected_labels"]         for x in batch], -100),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ORPOTrainer(Trainer):

    def __init__(self, beta: float, **kwargs: Any) -> None:
        self.orpo_beta = beta
        super().__init__(**kwargs)

    @staticmethod
    def _sequence_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Mean log-probability over non-masked response tokens.

        Args:
            logits: [B, T, V]
            labels: [B, T]  (-100 for masked positions)
        Returns:
            [B] mean log-prob per sequence
        """
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # [B, T-1, V]
        targets   = labels[:, 1:].clone()                      # [B, T-1]
        mask      = targets != -100                            # [B, T-1]

        targets[~mask] = 0                                     # safe index
        selected  = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
        selected  = selected * mask.float()

        return selected.sum(-1) / mask.float().sum(-1).clamp(min=1.0)      # [B]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def compute_loss(
        self,
        model: AutoModelForCausalLM,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple:
        # --- Chosen forward pass (provides SFT loss) ---
        chosen_out = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            labels=inputs["chosen_labels"],
        )
        sft_loss        = chosen_out.loss
        chosen_log_prob = self._sequence_log_probs(chosen_out.logits, inputs["chosen_labels"])

        # --- Rejected forward pass ---
        rejected_out = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            labels=inputs["rejected_labels"],
        )
        rejected_log_prob = self._sequence_log_probs(rejected_out.logits, inputs["rejected_labels"])

        # --- Odds-ratio loss ---
        def log_odds(log_p: torch.Tensor) -> torch.Tensor:
            # log(p / (1 - p)) = log_p - log(1 - exp(log_p))
            return log_p - torch.log(1.0 - torch.exp(log_p.clamp(max=-1e-7)) + 1e-7)

        log_or   = log_odds(chosen_log_prob) - log_odds(rejected_log_prob)
        or_loss  = -F.logsigmoid(log_or).mean()

        loss = sft_loss + self.orpo_beta * or_loss

        return (loss, chosen_out) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    parser = argparse.ArgumentParser(description="ORPO training without a reference model")
    parser.add_argument(
        "--model-name", default="gpt2",
        help="Base model name or local path",
    )
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_train.jsonl"))
    parser.add_argument("--eval-file",  type=Path, default=Path("data/processed/training/hh_rlhf/hh_rlhf_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/orpo"))
    parser.add_argument("--epochs",            type=int,   default=1)
    parser.add_argument("--batch-size",        type=int,   default=4)
    parser.add_argument("--grad-accum",        type=int,   default=4)
    parser.add_argument("--lr",                type=float, default=1e-5)
    parser.add_argument("--beta",              type=float, default=0.1, help="Odds-ratio penalty weight (λ)")
    parser.add_argument("--max-length",        type=int,   default=512)
    parser.add_argument("--max-prompt-length", type=int,   default=256)
    parser.add_argument("--max-samples",       type=int,   default=None)
    parser.add_argument("--fp16", action="store_true")
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

    tokenize = lambda ex: tokenize_row(ex, tokenizer, args.max_length, args.max_prompt_length)
    train_dataset = Dataset.from_list([tokenize(r) for r in train_rows])
    eval_dataset  = Dataset.from_list([tokenize(r) for r in eval_rows])

    print(f"Train examples: {len(train_dataset)} | Eval examples: {len(eval_dataset)}")

    config = ORPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=100,
        fp16=args.fp16,
        bf16=False,
        use_cpu=not torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = ORPOTrainer(
        beta=args.beta,
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ORPODataCollator(pad_token_id=tokenizer.pad_token_id),
    )

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"ORPO model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
