"""Preprocess Anthropic HH-RLHF into prompt/chosen/rejected JSONL files.

The original dataset stores each example as two full dialogue transcripts:
`chosen` and `rejected`. For SFT/DPO/ORPO training we split them into:

    prompt:   the shared dialogue prefix ending at the final "Assistant:"
    chosen:   the preferred assistant response
    rejected: the dispreferred assistant response
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset


ASSISTANT_MARKER = "Assistant:"


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving dialogue line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def split_preference_pair(example: dict[str, Any]) -> dict[str, str]:
    """Split HH-RLHF chosen/rejected full transcripts into a training triple."""
    chosen_full = normalize_text(example["chosen"])
    rejected_full = normalize_text(example["rejected"])

    prefix_end = common_prefix_len(chosen_full, rejected_full)
    shared_prefix = chosen_full[:prefix_end]

    # Move the boundary back to the last assistant turn so both responses are
    # complete candidate answers to the same prompt.
    assistant_pos = shared_prefix.rfind(ASSISTANT_MARKER)
    if assistant_pos == -1:
        prompt = shared_prefix.strip()
    else:
        prompt = shared_prefix[: assistant_pos + len(ASSISTANT_MARKER)].strip()

    chosen = chosen_full[len(prompt) :].strip()
    rejected = rejected_full[len(prompt) :].strip()

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def is_valid_example(example: dict[str, str]) -> bool:
    return bool(example["prompt"] and example["chosen"] and example["rejected"])


def save_jsonl(dataset: "Dataset", path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in dataset:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess(
    data_dir: str,
    cache_dir: Path,
    output_dir: Path,
    max_train_samples: int | None,
    max_test_samples: int | None,
) -> None:
    from datasets import DatasetDict, load_dataset

    raw = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir=data_dir,
        cache_dir=str(cache_dir),
    )
    if not isinstance(raw, DatasetDict):
        raise TypeError("Expected Anthropic/hh-rlhf to load as a DatasetDict.")

    processed = raw.map(
        split_preference_pair,
        remove_columns=raw["train"].column_names,
        desc="Splitting chosen/rejected transcripts",
    )
    processed = processed.filter(is_valid_example, desc="Dropping invalid rows")

    train = processed["train"]
    test = processed["test"]

    if max_train_samples is not None:
        train = train.select(range(min(max_train_samples, len(train))))
    if max_test_samples is not None:
        test = test.select(range(min(max_test_samples, len(test))))

    save_jsonl(train, output_dir / "hh_rlhf_train.jsonl")
    save_jsonl(test, output_dir / "hh_rlhf_test.jsonl")

    print(f"Saved train rows: {len(train):,} -> {output_dir / 'hh_rlhf_train.jsonl'}")
    print(f"Saved test rows:  {len(test):,} -> {output_dir / 'hh_rlhf_test.jsonl'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="harmless-base",
        help=(
            "HH-RLHF subset to load. For this project, 'harmless-base' is the "
            "best match for toxicity/safety alignment."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/training/hh_rlhf"),
        help="Directory for processed JSONL files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/huggingface"),
        help="Directory for Hugging Face downloaded dataset cache.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional small-sample cap for debugging.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional small-sample cap for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )


if __name__ == "__main__":
    main()
