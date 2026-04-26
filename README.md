# CS5788 Final Project

## Dataset Layout

The project separates training/validation datasets from the final evaluation
dataset:

```text
data/
  raw/
    huggingface/                         # Hugging Face download cache
  processed/
    training/
      hh_rlhf/
        hh_rlhf_train.jsonl              # train split for SFT/DPO/ORPO
        hh_rlhf_test.jsonl               # held-out validation split
    evaluation/
      realtoxicity/
        realtoxicity_all.jsonl           # final toxicity evaluation prompts
        realtoxicity_challenging.jsonl   # challenging=True prompts
```

HH-RLHF is used for model training and validation. RealToxicityPrompts is used
later for the final toxicity test in the proposal.

## HH-RLHF Preprocessing

The Anthropic HH-RLHF preprocessing script downloads the dataset from Hugging
Face and converts each example from full `chosen` / `rejected` dialogue
transcripts into:

```json
{
  "prompt": "Human: ...\nAssistant:",
  "chosen": "preferred assistant response",
  "rejected": "dispreferred assistant response"
}
```

## Execution Order

### 1. Install Dependencies

Run this from the project root:

```bash
pip install -r requirements.txt
```

The first dataset run requires internet access because Hugging Face will download
`Anthropic/hh-rlhf`.

By default, the script stores the Hugging Face download cache inside this
project:

```text
data/raw/huggingface
```

This avoids using the default Windows cache location on the C drive.

### 2. Run a Small HH-RLHF Test

Before processing the full dataset, run a small sample:

```bash
python scripts/preprocess_hh_rlhf.py --max-train-samples 100 --max-test-samples 20
```

This checks that downloading, parsing, and writing files all work.

### 3. Check the HH-RLHF Output Files

After the test run, confirm these files exist:

```text
data/processed/training/hh_rlhf/hh_rlhf_train.jsonl
data/processed/training/hh_rlhf/hh_rlhf_test.jsonl
```

Each line should be one JSON object with `prompt`, `chosen`, and `rejected`.

### 4. Run Full HH-RLHF Preprocessing

After the small test succeeds, run the full preprocessing job:

```bash
python scripts/preprocess_hh_rlhf.py
```

By default, this uses the `harmless-base` subset:

```text
Anthropic/hh-rlhf, data_dir="harmless-base"
```

This subset is the best match for the project goal of toxicity and safety
alignment.

To choose another cache location, pass `--cache-dir`:

```bash
python scripts/preprocess_hh_rlhf.py --cache-dir data/raw/huggingface
```

### 5. Optional: Use Another HH-RLHF Subset

To process a different subset, pass `--data-dir`:

```bash
python scripts/preprocess_hh_rlhf.py --data-dir helpful-base
```

For this project, prefer `harmless-base` unless there is a specific reason to
include helpfulness data.

## How the Processed HH-RLHF Files Are Used

```text
SFT:  use prompt + chosen
DPO:  use prompt + chosen + rejected
ORPO: use prompt + chosen + rejected
```

The HH-RLHF `test` split is a held-out validation split for training. It is not
the final toxicity evaluation dataset. The final toxicity evaluation should use
RealToxicityPrompts under:

```text
data/processed/evaluation/realtoxicity/
```

## RealToxicityPrompts Preprocessing

RealToxicityPrompts is used for the final toxicity evaluation, not for training.
The script extracts prompt text and toxicity metadata into:

```json
{
  "filename": "...",
  "begin": 0,
  "end": 0,
  "challenging": true,
  "prompt": "evaluation prompt text",
  "continuation": "original dataset continuation",
  "prompt_toxicity": 0.0,
  "continuation_toxicity": 0.0
}
```

### 1. Run a Small RealToxicityPrompts Test

```bash
python scripts/preprocess_realtoxicity.py --max-samples 100
```

### 2. Check the RealToxicityPrompts Output Files

```text
data/processed/evaluation/realtoxicity/realtoxicity_all.jsonl
data/processed/evaluation/realtoxicity/realtoxicity_challenging.jsonl
```

For final evaluation, prefer:

```text
data/processed/evaluation/realtoxicity/realtoxicity_challenging.jsonl
```

because the proposal prioritizes `challenging=True` prompts.

### 3. Run Full RealToxicityPrompts Preprocessing

```bash
python scripts/preprocess_realtoxicity.py
```
