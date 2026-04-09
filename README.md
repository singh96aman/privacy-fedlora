# FedLoRA-KD

Privacy-preserving federated learning with Low-Rank Adapters and Knowledge Distillation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test
python scripts/smoke_test.py

# Run full experiment
python main.py --config configs/fedlora_squad_nq.json

# Run specific phase only
python main.py --config configs/fedlora_squad_nq.json --phase baseline
python main.py --config configs/fedlora_squad_nq.json --phase train
```

## Project Structure

```
├── main.py                      # Entry point
├── configs/
│   └── fedlora_squad_nq.json    # Main experiment config
├── src/
│   ├── model.py                 # Model loading, LoRA setup
│   ├── data.py                  # Dataset loading, preprocessing
│   ├── trainer.py               # Training loop
│   ├── evaluator.py             # QA metrics (F1, EM)
│   ├── aggregator.py            # FedAvg aggregation
│   ├── attacks.py               # Privacy attacks
│   ├── client.py                # FL client
│   └── server.py                # FL server
├── tests/                       # Unit tests
├── scripts/                     # Utility scripts
└── outputs/                     # Experiment results
```

## Experiment Phases

1. **Baseline**: Evaluate base Llama 3.2 3B on SQuAD and Natural Questions
2. **Client Training**: Train LoRA adapters for each client (C1: SQuAD, C2: NQ)
3. **Aggregation**: Create Universal Adapter via FedAvg
4. **Evaluation**: Evaluate Universal Model on both datasets
5. **Privacy Analysis**: Run membership inference and domain identification attacks

## Requirements

- NVIDIA GPU with 24GB+ VRAM (tested on A10)
- Python 3.10+
- HuggingFace account with Llama access

## Running Tests

```bash
pytest tests/ -v
```
