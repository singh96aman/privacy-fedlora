#!/usr/bin/env python
"""Quick smoke test to verify all imports work."""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        import torch
        print(f"  torch: {torch.__version__}")
    except ImportError as e:
        print(f"  torch: FAILED - {e}")
        return False

    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"  transformers: FAILED - {e}")
        return False

    try:
        import peft
        print(f"  peft: {peft.__version__}")
    except ImportError as e:
        print(f"  peft: FAILED - {e}")
        return False

    try:
        import datasets
        print(f"  datasets: {datasets.__version__}")
    except ImportError as e:
        print(f"  datasets: FAILED - {e}")
        return False

    try:
        import accelerate
        print(f"  accelerate: {accelerate.__version__}")
    except ImportError as e:
        print(f"  accelerate: FAILED - {e}")
        return False

    # Test our modules
    print("\nTesting project modules...")
    try:
        from src import model, data, trainer, evaluator, aggregator, attacks
        print("  src modules: OK")
    except ImportError as e:
        print(f"  src modules: FAILED - {e}")
        return False

    try:
        from src import Client, Server
        print("  Client, Server: OK")
    except ImportError as e:
        print(f"  Client, Server: FAILED - {e}")
        return False

    print("\nAll imports successful!")
    return True


def test_gpu():
    """Test GPU availability."""
    import torch
    print(f"\nGPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def test_config():
    """Test config loading."""
    import json
    from pathlib import Path

    config_path = Path("configs/fedlora_squad_nq.json")
    print(f"\nConfig file exists: {config_path.exists()}")

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"Experiment name: {config['experiment_name']}")
        print(f"Model: {config['model']['name']}")
        print(f"Clients: {list(config['clients'].keys())}")


if __name__ == "__main__":
    print("=" * 50)
    print("FedLoRA-KD Smoke Test")
    print("=" * 50)

    success = test_imports()
    if success:
        test_gpu()
        test_config()
        sys.exit(0)
    else:
        print("\nSmoke test FAILED. Install dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
