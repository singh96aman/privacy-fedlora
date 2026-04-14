#!/usr/bin/env python
"""Run all C3 experiments to fill the comparison table.

Experiments:
1. BM - Base Model on C3
2. UM - Universal Model (C1+C2) on C3
3. BM + (C3) - Base Model fine-tuned on C3
4. BM + (C3 w UM KD) - Fine-tuned on C3 with dual-teacher KD
5. BM + AVG(C1, C2, C3) - Universal Model v2
"""

import argparse
import json
import sys
from pathlib import Path

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import load_base_model, setup_lora, load_adapter, save_adapter
from src.data import (
    load_sciq, format_sciq_example, preprocess_dataset, create_dataloader
)
from src.evaluator import evaluate
from src.trainer import train_lora
from src.kd_trainer import train_with_confidence_weighted_kd
from src.aggregator import fedavg_lora, load_adapter_weights, save_aggregated_adapter


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


def save_results(results: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


def get_c3_data(tokenizer, config, num_train=5000, num_eval=500):
    """Load and preprocess C3 data from config."""
    from src.data import (load_sciq, format_sciq_example,
        load_billsum, format_billsum_example,
        load_samsum, format_samsum_example)

    c3_domain = config["c3_domains"][0]
    domain_name = c3_domain["dataset"]

    registry = {
        "sciq": (load_sciq, "validation", format_sciq_example),
        "billsum": (load_billsum, "test", format_billsum_example),
        "samsum": (load_samsum, "test", format_samsum_example),
    }

    load_fn, eval_split, format_fn = registry[domain_name]
    print(f"\n>>> C3 Domain: {domain_name} (train: train, eval: {eval_split})")
    train_data = load_fn("train", num_train)
    eval_data = load_fn(eval_split, num_eval)

    max_length = config["training"].get("max_seq_length", 512)
    train_dataset = preprocess_dataset(train_data, tokenizer, domain_name, max_length)
    eval_dataset = preprocess_dataset(eval_data, tokenizer, domain_name, max_length)
    eval_examples = [format_fn(ex) for ex in eval_data]

    return train_dataset, eval_dataset, eval_examples


def experiment_bm(config: dict, output_dir: Path):
    """Experiment 1: Base Model on C3."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Base Model (BM) on C3")
    print("=" * 60)

    model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )

    _, _, eval_examples = get_c3_data(tokenizer, config)

    results = evaluate(
        model, tokenizer, eval_examples,
        max_samples=500,
        compute_all_metrics=True
    )

    save_results(results, output_dir / "bm_c3.json")
    print(f"BM Results: F1={results['f1']:.4f}, EM={results['exact_match']:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def experiment_um(config: dict, output_dir: Path, um_adapter_path: str):
    """Experiment 2: Universal Model (C1+C2) on C3."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Universal Model (UM) on C3")
    print("=" * 60)

    model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )

    # Load universal adapter
    model = load_adapter(model, um_adapter_path)
    model.eval()

    _, _, eval_examples = get_c3_data(tokenizer, config)

    results = evaluate(
        model, tokenizer, eval_examples,
        max_samples=500,
        compute_all_metrics=True
    )

    save_results(results, output_dir / "um_c3.json")
    print(f"UM Results: F1={results['f1']:.4f}, EM={results['exact_match']:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def experiment_bm_c3(config: dict, output_dir: Path):
    """Experiment 3: Base Model + LoRA trained on C3 only."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: BM + (C3)")
    print("=" * 60)

    model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=True
    )

    # Setup LoRA
    lora_config = config.get("lora", {})
    model = setup_lora(model, lora_config)

    # Get C3 data
    train_dataset, eval_dataset, eval_examples = get_c3_data(tokenizer, config)

    batch_size = config["training"].get("batch_size", 4)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size)
    eval_loader = create_dataloader(eval_dataset, batch_size=batch_size, shuffle=False)

    # Train
    train_metrics = train_lora(model, train_loader, config, eval_loader)

    # Save adapter
    adapter_path = output_dir / "bm_c3_adapter"
    save_adapter(model, str(adapter_path))

    # Evaluate
    model.eval()
    results = evaluate(
        model, tokenizer, eval_examples,
        max_samples=500,
        compute_all_metrics=True
    )
    results["train_loss"] = train_metrics["train_loss"]

    save_results(results, output_dir / "bm_c3_finetuned.json")
    print(f"BM+C3 Results: F1={results['f1']:.4f}, EM={results['exact_match']:.4f}")

    del model
    torch.cuda.empty_cache()
    return results, str(adapter_path)


def experiment_bm_c3_kd(config: dict, output_dir: Path, um_adapter_path: str):
    """Experiment 4: BM + C3 with confidence-weighted KD from UM."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: BM + (C3 w UM KD) - Confidence Weighted")
    print("=" * 60)

    # Load universal teacher (UM only - no base teacher)
    teacher, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )
    teacher = load_adapter(teacher, um_adapter_path)

    # Create student (fresh LoRA on base)
    student, _ = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=True
    )
    lora_config = config.get("lora", {})
    student = setup_lora(student, lora_config)

    # Get C3 data
    train_dataset, eval_dataset, eval_examples = get_c3_data(tokenizer, config)

    batch_size = config["training"].get("batch_size", 4)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size)

    # Train with confidence-weighted KD (single teacher: UM)
    train_metrics = train_with_confidence_weighted_kd(
        student, teacher, train_loader, config
    )

    # Save adapter
    adapter_path = output_dir / "bm_c3_kd_adapter"
    save_adapter(student, str(adapter_path))

    # Evaluate
    student.eval()
    results = evaluate(
        student, tokenizer, eval_examples,
        max_samples=500,
        compute_all_metrics=True
    )
    results["train_loss"] = train_metrics["train_loss"]
    results["avg_confidence"] = train_metrics["avg_confidence"]

    save_results(results, output_dir / "bm_c3_kd.json")
    print(f"BM+C3(KD) Results: F1={results['f1']:.4f}, "
          f"EM={results['exact_match']:.4f}, "
          f"Avg Confidence={results['avg_confidence']:.3f}")

    del student, teacher
    torch.cuda.empty_cache()
    return results, str(adapter_path)


def experiment_um_v2(config: dict, output_dir: Path, c1_path: str, c2_path: str, c3_path: str):
    """Experiment 5: Universal Model v2 = AVG(C1, C2, C3)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: BM + AVG(C1, C2, C3)")
    print("=" * 60)

    # Load all adapters
    c1_weights = load_adapter_weights(c1_path)
    c2_weights = load_adapter_weights(c2_path)
    c3_weights = load_adapter_weights(c3_path)

    # Aggregate
    aggregated = fedavg_lora([c1_weights, c2_weights, c3_weights])

    # Save
    um_v2_path = output_dir / "um_v2_adapter"
    save_aggregated_adapter(aggregated, str(um_v2_path), c1_path)

    # Load and evaluate
    model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )
    model = load_adapter(model, str(um_v2_path))
    model.eval()

    _, _, eval_examples = get_c3_data(tokenizer, config)

    results = evaluate(
        model, tokenizer, eval_examples,
        max_samples=500,
        compute_all_metrics=True
    )

    save_results(results, output_dir / "um_v2_c3.json")
    print(f"UM_v2 Results: F1={results['f1']:.4f}, EM={results['exact_match']:.4f}")

    del model
    torch.cuda.empty_cache()
    return results


def print_comparison_table(results: dict):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("COMPARISON TABLE (C3)")
    print("=" * 100)

    headers = ["Model", "F1", "EM", "Contains", "PPL", "BERTScore", "BLEU", "ROUGE"]
    print(f"{'Model':<25} {'F1':>8} {'EM':>8} {'Contains':>10} {'PPL':>10} {'BERT-F1':>10} {'BLEU':>8} {'ROUGE':>8}")
    print("-" * 100)

    for name, r in results.items():
        print(f"{name:<25} "
              f"{r.get('f1', 0):>8.4f} "
              f"{r.get('exact_match', 0):>8.4f} "
              f"{r.get('contains', 0):>10.4f} "
              f"{r.get('perplexity', 0):>10.2f} "
              f"{r.get('bertscore_f1', 0):>10.4f} "
              f"{r.get('bleu', 0):>8.4f} "
              f"{r.get('rouge_l', 0):>8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run C3 experiments")
    parser.add_argument("--config", type=str, default="configs/fedlora_squad_triviaqa.json")
    parser.add_argument("--output-dir", type=str, default="outputs/c3_experiments")
    parser.add_argument("--um-adapter", type=str, required=True,
                        help="Path to Universal Model adapter (from C1+C2 training)")
    parser.add_argument("--c1-adapter", type=str, required=True,
                        help="Path to C1 adapter")
    parser.add_argument("--c2-adapter", type=str, required=True,
                        help="Path to C2 adapter")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "bm", "um", "bm_c3", "bm_c3_kd", "um_v2"])
    args = parser.parse_args()

    set_seed(42)

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.experiment in ["all", "bm"]:
        all_results["BM"] = experiment_bm(config, output_dir)

    if args.experiment in ["all", "um"]:
        all_results["UM"] = experiment_um(config, output_dir, args.um_adapter)

    if args.experiment in ["all", "bm_c3"]:
        results, c3_path = experiment_bm_c3(config, output_dir)
        all_results["BM + (C3)"] = results

    if args.experiment in ["all", "bm_c3_kd"]:
        results, c3_kd_path = experiment_bm_c3_kd(config, output_dir, args.um_adapter)
        all_results["BM + (C3 w UM KD)"] = results

    if args.experiment in ["all", "um_v2"]:
        # Use C3 adapter from bm_c3 experiment
        c3_adapter = output_dir / "bm_c3_adapter"
        if c3_adapter.exists():
            all_results["BM + AVG(C1,C2,C3)"] = experiment_um_v2(
                config, output_dir,
                args.c1_adapter, args.c2_adapter, str(c3_adapter)
            )

    if all_results:
        print_comparison_table(all_results)
        save_results(all_results, output_dir / "comparison_table.json")


if __name__ == "__main__":
    main()
