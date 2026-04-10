#!/usr/bin/env python
"""Modular training script for FedLoRA-KD.

Run individual stages:
    python scripts/train.py --config configs/fedlora.json --stage c1_adapter
    python scripts/train.py --config configs/fedlora.json --stage c2_adapter
    python scripts/train.py --config configs/fedlora.json --stage c3_adapter
    python scripts/train.py --config configs/fedlora.json --stage aggregate --adapters c1,c2
    python scripts/train.py --config configs/fedlora.json --stage evaluate --model um

Override base model:
    python scripts/train.py --config configs/fedlora.json --bm meta-llama/Llama-3.2-3B --stage c1_adapter
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def save_metrics(metrics: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {path}")


def train_adapter(config: dict, client_id: str, output_dir: Path, base_model: str = None):
    """Train a single client adapter."""
    from src.model import load_base_model, setup_lora, save_adapter
    from src.data import get_client_data, create_dataloader
    from src.trainer import train_lora

    print(f"\n{'='*60}")
    print(f"Training adapter: {client_id}")
    print(f"{'='*60}")

    # Override base model if specified
    model_name = base_model or config["model"]["name"]
    print(f"Base model: {model_name}")

    model, tokenizer = load_base_model(
        model_name=model_name,
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=config["model"].get("gradient_checkpointing", True)
    )

    # Setup LoRA
    lora_config = config.get("lora", {})
    model = setup_lora(model, lora_config)

    # Load client data
    train_dataset, eval_dataset = get_client_data(client_id, config, tokenizer)
    print(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    batch_size = config["training"].get("batch_size", 4)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size)
    eval_loader = create_dataloader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Train
    metrics = train_lora(model, train_loader, config, eval_loader)

    # Save
    adapter_path = output_dir / f"{client_id}_adapter"
    save_adapter(model, str(adapter_path))
    save_metrics(metrics, output_dir / f"{client_id}_metrics.json")

    print(f"Adapter saved: {adapter_path}")
    print(f"Train loss: {metrics['train_loss']:.4f}")

    del model
    torch.cuda.empty_cache()

    return str(adapter_path)


def aggregate_adapters(config: dict, adapter_ids: list, output_dir: Path, output_name: str = "universal"):
    """Aggregate multiple adapters into one."""
    from src.aggregator import load_adapter_weights, fedavg_lora, save_aggregated_adapter

    print(f"\n{'='*60}")
    print(f"Aggregating adapters: {adapter_ids}")
    print(f"{'='*60}")

    # Load adapters
    adapter_dicts = []
    first_path = None

    for adapter_id in adapter_ids:
        path = output_dir / f"{adapter_id}_adapter"
        if not path.exists():
            raise FileNotFoundError(f"Adapter not found: {path}")

        weights = load_adapter_weights(str(path))
        adapter_dicts.append(weights)

        if first_path is None:
            first_path = str(path)

        print(f"Loaded: {path}")

    # Aggregate
    aggregated = fedavg_lora(adapter_dicts)

    # Save
    output_path = output_dir / f"{output_name}_adapter"
    save_aggregated_adapter(aggregated, str(output_path), first_path)

    print(f"Aggregated adapter saved: {output_path}")

    return str(output_path)


def evaluate_model(
    config: dict,
    output_dir: Path,
    model_type: str,
    adapter_path: str = None,
    dataset: str = None,
    base_model: str = None
):
    """Evaluate a model configuration."""
    from src.model import load_base_model, load_adapter
    from src.data import (
        load_squad, load_sciq,
        format_squad_example, format_sciq_example
    )
    from src.evaluator import evaluate_qa

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_type}")
    print(f"{'='*60}")

    model_name = base_model or config["model"]["name"]

    model, tokenizer = load_base_model(
        model_name=model_name,
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )

    # Load adapter if specified
    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = load_adapter(model, adapter_path)

    model.eval()

    # Determine dataset
    eval_samples = config["evaluation"].get("eval_samples", 500)

    dataset_loaders = {
        "squad": (load_squad, format_squad_example, "validation"),
        "sciq": (load_sciq, format_sciq_example, "validation"),
    }

    results = {}

    datasets_to_eval = [dataset] if dataset else list(dataset_loaders.keys())

    for ds_name in datasets_to_eval:
        if ds_name not in dataset_loaders:
            print(f"Unknown dataset: {ds_name}")
            continue

        loader_fn, format_fn, split = dataset_loaders[ds_name]
        print(f"\nEvaluating on {ds_name}...")

        raw_data = loader_fn(split, eval_samples)
        examples = [format_fn(ex) for ex in raw_data]

        ds_results = evaluate_qa(
            model, tokenizer, examples,
            max_samples=eval_samples,
            compute_all_metrics=True
        )

        results[ds_name] = ds_results
        print(f"{ds_name}: F1={ds_results['f1']:.4f}, EM={ds_results['exact_match']:.4f}")

    # Save
    save_metrics(results, output_dir / f"{model_type}_eval.json")

    del model
    torch.cuda.empty_cache()

    return results


def run_baseline(config: dict, output_dir: Path, base_model: str = None):
    """Evaluate base model without any adapters."""
    return evaluate_model(config, output_dir, "baseline", base_model=base_model)


def main():
    parser = argparse.ArgumentParser(
        description="FedLoRA-KD Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train C1 adapter
  python scripts/train.py --config configs/fedlora.json --stage c1_adapter

  # Train with different base model
  python scripts/train.py --config configs/fedlora.json --bm meta-llama/Llama-3.2-1B --stage c1_adapter

  # Aggregate C1 and C2 into Universal Model
  python scripts/train.py --config configs/fedlora.json --stage aggregate --adapters c1,c2

  # Evaluate Universal Model on SciQ
  python scripts/train.py --config configs/fedlora.json --stage evaluate --model um --dataset sciq
        """
    )

    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--bm", type=str, help="Override base model (e.g., meta-llama/Llama-3.2-3B)")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument(
        "--stage", type=str, required=True,
        choices=[
            "c1_adapter", "c2_adapter", "c3_adapter",
            "aggregate", "evaluate", "baseline", "all"
        ],
        help="Stage to run"
    )
    parser.add_argument("--adapters", type=str, help="Comma-separated adapter IDs for aggregation (e.g., c1,c2)")
    parser.add_argument("--adapter-path", type=str, help="Path to adapter for evaluation")
    parser.add_argument("--model", type=str, help="Model name for evaluation output")
    parser.add_argument("--dataset", type=str, help="Dataset for evaluation (squad, nq, sciq)")
    parser.add_argument("--output-name", type=str, default="universal", help="Name for aggregated adapter")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["logging"]["output_dir"]) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Base model: {args.bm or config['model']['name']}")

    # Execute stage
    if args.stage == "c1_adapter":
        train_adapter(config, "c1", output_dir, args.bm)

    elif args.stage == "c2_adapter":
        train_adapter(config, "c2", output_dir, args.bm)

    elif args.stage == "c3_adapter":
        train_adapter(config, "c3", output_dir, args.bm)

    elif args.stage == "aggregate":
        if not args.adapters:
            parser.error("--adapters required for aggregate stage")
        adapter_ids = [a.strip() for a in args.adapters.split(",")]
        aggregate_adapters(config, adapter_ids, output_dir, args.output_name)

    elif args.stage == "evaluate":
        adapter_path = args.adapter_path
        if not adapter_path and args.model:
            # Try to find adapter in output dir
            potential_path = output_dir / f"{args.model}_adapter"
            if potential_path.exists():
                adapter_path = str(potential_path)

        evaluate_model(
            config, output_dir,
            model_type=args.model or "model",
            adapter_path=adapter_path,
            dataset=args.dataset,
            base_model=args.bm
        )

    elif args.stage == "baseline":
        run_baseline(config, output_dir, args.bm)

    elif args.stage == "all":
        # Run full pipeline
        print("\n" + "="*60)
        print("Running full pipeline")
        print("="*60)

        # Baseline
        run_baseline(config, output_dir, args.bm)

        # Train adapters
        for client_id in config["clients"].keys():
            train_adapter(config, client_id, output_dir, args.bm)

        # Aggregate C1 + C2
        aggregate_adapters(config, ["c1", "c2"], output_dir, "universal")

        # Evaluate
        evaluate_model(config, output_dir, "universal", str(output_dir / "universal_adapter"), base_model=args.bm)

    print("\nDone!")


if __name__ == "__main__":
    main()
