# FedLoRA-KD

Privacy-preserving federated learning with Low-Rank Adapters and Knowledge Distillation.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure HuggingFace token
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your HuggingFace token
# HF_TOKEN=hf_xxxxxxxxxxxxx
```

Or login via CLI:
```bash
huggingface-cli login
```

### 3. Verify setup
```bash
python scripts/smoke_test.py
```

---

## Training Adapters

### Train individual client adapters

```bash
# Train C1 adapter (SQuAD)
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage c1_adapter

# Train C2 adapter (TriviaQA)
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage c2_adapter

# Train C3 adapter (SciQ)
python scripts/train.py --config configs/c3_experiment.json --stage c3_adapter
```

### Override base model
```bash
# Use a different model
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --bm meta-llama/Llama-3.2-1B --stage c1_adapter
```

### Aggregate adapters into Universal Model
```bash
# Create UM from C1 + C2
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage aggregate --adapters c1,c2

# Create UM v2 from C1 + C2 + C3
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage aggregate --adapters c1,c2,c3 --output-name universal_v2
```

### Evaluate models
```bash
# Evaluate baseline (no adapter)
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage baseline

# Evaluate Universal Model on specific dataset
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage evaluate --model universal --dataset sciq

# Evaluate with custom adapter path
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage evaluate --adapter-path outputs/my_adapter --model custom
```

### Run full pipeline
```bash
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage all
```

---

## C3 Experiments (Comparison Table)

To fill the comparison table:

| Model | Description |
|-------|-------------|
| BM | Base Model (no adapter) |
| UM | Universal Model (C1+C2) |
| BM + (C3) | Fine-tuned on C3 only |
| BM + (C3 w UM KD) | Dual-teacher KD from UM |
| BM + AVG(C1,C2,C3) | Universal Model v2 |

### Step-by-step

```bash
# 1. Train C1 and C2 adapters
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage c1_adapter
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage c2_adapter

# 2. Create Universal Model (C1+C2)
python scripts/train.py --config configs/fedlora_squad_triviaqa.json --stage aggregate --adapters c1,c2

# 3. Run C3 experiments
python scripts/run_c3_experiments.py \
    --config configs/c3_experiment.json \
    --output-dir outputs/c3_experiments \
    --um-adapter outputs/fedlora_squad_triviaqa/universal_adapter \
    --c1-adapter outputs/fedlora_squad_triviaqa/c1_adapter \
    --c2-adapter outputs/fedlora_squad_triviaqa/c2_adapter
```

Results saved to: `outputs/c3_experiments/comparison_table.json`

---

## Project Structure

```
├── main.py                      # Legacy entry point
├── scripts/
│   ├── train.py                 # Modular training script
│   ├── run_c3_experiments.py    # C3 comparison experiments
│   └── smoke_test.py            # Verify setup
├── configs/
│   ├── fedlora_squad_triviaqa.json    # C1 + C2 config
│   └── c3_experiment.json       # C3 experiment config
├── src/
│   ├── model.py                 # Model loading, LoRA setup
│   ├── data.py                  # Dataset loading (SQuAD, NQ, SciQ)
│   ├── trainer.py               # Training loop
│   ├── kd_trainer.py            # Knowledge distillation
│   ├── evaluator.py             # Metrics (F1, EM, BLEU, etc.)
│   ├── aggregator.py            # FedAvg aggregation
│   ├── attacks.py               # Privacy attacks
│   ├── client.py                # FL client
│   └── server.py                # FL server
├── tests/                       # Unit tests
└── outputs/                     # Experiment results
```

---

## Configuration

Adapters are defined in the config file:

```json
{
    "clients": {
        "c1": {"dataset": "squad_v2", "num_samples": 10000},
        "c2": {"dataset": "triviaqa", "num_samples": 10000},
        "c3": {"dataset": "sciq", "num_samples": 5000}
    }
}
```

Available datasets: `squad_v2`, `triviaqa`, `natural_questions`, `sciq`

---

## Requirements

- NVIDIA GPU with 24GB+ VRAM (tested on A10)
- Python 3.10+
- HuggingFace account with Llama access

---

## Running Tests

```bash
pytest tests/ -v
```
