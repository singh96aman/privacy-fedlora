"""Main entry point for FedLoRA-KD experiments."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> Dict:
    """Load and validate experiment configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path) as f:
        config = json.load(f)

    required = ["experiment_name", "seed", "model", "clients", "training"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    return config


def save_metrics(metrics: Dict, path: str) -> None:
    """Save metrics to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {path}")


def run_baseline_evaluation(config: Dict, output_dir: Path) -> Dict:
    """Evaluate base model without any adapters."""
    from src.model import load_base_model
    from src.data import load_squad, load_natural_questions, format_squad_example, format_nq_example
    from src.evaluator import evaluate_qa

    print("\n" + "=" * 50)
    print("PHASE 1: Baseline Evaluation")
    print("=" * 50)

    model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False  # Not needed for inference
    )

    eval_samples = config["evaluation"].get("eval_samples", 500)
    baseline_metrics = {}

    # Evaluate on SQuAD
    print("\nEvaluating on SQuAD 2.0...")
    squad_data = load_squad("validation", eval_samples)
    squad_examples = [format_squad_example(ex) for ex in squad_data]
    baseline_metrics["squad"] = evaluate_qa(model, tokenizer, squad_examples, eval_samples)
    print(f"SQuAD F1: {baseline_metrics['squad']['f1']:.4f}, EM: {baseline_metrics['squad']['exact_match']:.4f}")

    # Evaluate on NQ
    print("\nEvaluating on Natural Questions...")
    nq_data = load_natural_questions("validation", eval_samples)
    nq_examples = [format_nq_example(ex) for ex in nq_data]
    baseline_metrics["nq"] = evaluate_qa(model, tokenizer, nq_examples, eval_samples)
    print(f"NQ F1: {baseline_metrics['nq']['f1']:.4f}, EM: {baseline_metrics['nq']['exact_match']:.4f}")

    save_metrics(baseline_metrics, output_dir / "baseline" / "metrics.json")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return baseline_metrics


def run_client_training(config: Dict, output_dir: Path) -> Dict[str, str]:
    """Train LoRA adapters for each client."""
    from src.client import Client

    print("\n" + "=" * 50)
    print("PHASE 2: Client Training")
    print("=" * 50)

    adapter_paths = {}

    for client_id in config["clients"].keys():
        print(f"\n--- Training Client {client_id} ---")

        client = Client(client_id, config)
        client.setup()

        metrics = client.train()
        adapter_path = client.save(str(output_dir))

        save_metrics(
            {"training": metrics},
            output_dir / f"client_{client_id}" / "metrics.json"
        )

        adapter_paths[client_id] = adapter_path

        # Clean up
        del client
        torch.cuda.empty_cache()

    return adapter_paths


def run_aggregation(config: Dict, adapter_paths: Dict[str, str], output_dir: Path) -> str:
    """Aggregate client adapters into universal adapter."""
    from src.server import Server
    from src.aggregator import load_adapter_weights

    print("\n" + "=" * 50)
    print("PHASE 3: Server Aggregation")
    print("=" * 50)

    server = Server(config)

    # Load and send client updates
    for client_id, path in adapter_paths.items():
        weights = load_adapter_weights(path)
        server.receive_update(client_id, weights)

    # Aggregate
    server.aggregate()

    # Save universal adapter
    first_adapter_path = list(adapter_paths.values())[0]
    universal_path = server.save_universal_adapter(str(output_dir), first_adapter_path)

    return universal_path


def run_universal_evaluation(config: Dict, universal_path: str, output_dir: Path) -> Dict:
    """Evaluate universal model on all datasets."""
    from src.model import load_base_model, load_adapter
    from src.data import load_squad, load_natural_questions, format_squad_example, format_nq_example
    from src.evaluator import evaluate_qa

    print("\n" + "=" * 50)
    print("PHASE 4: Universal Model Evaluation")
    print("=" * 50)

    model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )

    # Load universal adapter
    model = load_adapter(model, universal_path)
    model.eval()

    eval_samples = config["evaluation"].get("eval_samples", 500)
    universal_metrics = {}

    # Evaluate on SQuAD
    print("\nEvaluating Universal Model on SQuAD 2.0...")
    squad_data = load_squad("validation", eval_samples)
    squad_examples = [format_squad_example(ex) for ex in squad_data]
    universal_metrics["squad"] = evaluate_qa(model, tokenizer, squad_examples, eval_samples)
    print(f"SQuAD F1: {universal_metrics['squad']['f1']:.4f}, EM: {universal_metrics['squad']['exact_match']:.4f}")

    # Evaluate on NQ
    print("\nEvaluating Universal Model on Natural Questions...")
    nq_data = load_natural_questions("validation", eval_samples)
    nq_examples = [format_nq_example(ex) for ex in nq_data]
    universal_metrics["nq"] = evaluate_qa(model, tokenizer, nq_examples, eval_samples)
    print(f"NQ F1: {universal_metrics['nq']['f1']:.4f}, EM: {universal_metrics['nq']['exact_match']:.4f}")

    save_metrics(universal_metrics, output_dir / "universal" / "metrics.json")

    del model
    torch.cuda.empty_cache()

    return universal_metrics


def run_privacy_analysis(config: Dict, adapter_paths: Dict[str, str], output_dir: Path) -> Dict:
    """Run privacy attack analysis."""
    from src.model import load_base_model, load_adapter
    from src.data import load_squad, load_natural_questions, preprocess_dataset, create_dataloader
    from src.evaluator import get_loss_distribution
    from src.attacks import membership_inference_attack, domain_identification_attack
    from src.aggregator import load_adapter_weights

    print("\n" + "=" * 50)
    print("PHASE 5: Privacy Analysis")
    print("=" * 50)

    privacy_metrics = {}

    # Load model
    model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )

    # Load universal adapter
    universal_path = output_dir / "universal" / "aggregated_adapter"
    model = load_adapter(model, str(universal_path))
    model.eval()

    # Membership Inference Attack
    print("\n--- Membership Inference Attack ---")
    num_samples = config["privacy"].get("num_shadow_samples", 1000)

    # Get member data (training data)
    squad_train = load_squad("train", num_samples)
    squad_train = preprocess_dataset(squad_train, tokenizer, "squad")
    member_loader = create_dataloader(squad_train, batch_size=8, shuffle=False)
    member_losses = get_loss_distribution(model, member_loader)

    # Get non-member data (validation data not seen during training)
    squad_val = load_squad("validation", num_samples)
    squad_val = preprocess_dataset(squad_val, tokenizer, "squad")
    non_member_loader = create_dataloader(squad_val, batch_size=8, shuffle=False)
    non_member_losses = get_loss_distribution(model, non_member_loader)

    mia_results = membership_inference_attack(member_losses, non_member_losses)
    privacy_metrics["membership_inference"] = mia_results
    print(f"MIA Accuracy: {mia_results['accuracy']:.4f} (baseline: {mia_results['baseline_accuracy']:.4f})")
    print(f"MIA AUC: {mia_results['auc']:.4f}")

    # Domain Identification Attack
    print("\n--- Domain Identification Attack ---")
    adapter_weights = {}
    labels = []
    for client_id, path in adapter_paths.items():
        weights = load_adapter_weights(path)
        adapter_weights[client_id] = weights
        labels.append(client_id)

    # Simple domain ID based on adapter statistics
    dia_results = domain_identification_attack(adapter_weights, labels)
    privacy_metrics["domain_identification"] = dia_results
    print(f"Domain ID Accuracy: {dia_results['accuracy']:.4f} (random: {dia_results['random_baseline']:.4f})")

    save_metrics(privacy_metrics, output_dir / "privacy" / "metrics.json")

    return privacy_metrics


def print_summary(baseline: Dict, universal: Dict, privacy: Dict) -> None:
    """Print experiment summary."""
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)

    print("\n--- Performance Comparison ---")
    print(f"{'Dataset':<15} {'Baseline F1':<15} {'Universal F1':<15} {'Improvement':<15}")
    print("-" * 60)

    for dataset in ["squad", "nq"]:
        base_f1 = baseline.get(dataset, {}).get("f1", 0)
        uni_f1 = universal.get(dataset, {}).get("f1", 0)
        improvement = uni_f1 - base_f1
        print(f"{dataset:<15} {base_f1:<15.4f} {uni_f1:<15.4f} {improvement:+.4f}")

    print("\n--- Privacy Leakage ---")
    mia = privacy.get("membership_inference", {})
    print(f"Membership Inference Attack AUC: {mia.get('auc', 0):.4f}")
    print(f"  (>0.5 indicates leakage, 1.0 = complete leakage)")

    dia = privacy.get("domain_identification", {})
    print(f"Domain Identification Accuracy: {dia.get('accuracy', 0):.4f}")
    print(f"  (>{dia.get('random_baseline', 0):.2f} indicates domain leakage)")


def main(config: Dict) -> None:
    """Run full FedLoRA-KD experiment."""
    set_seed(config["seed"])

    output_dir = Path(config["logging"]["output_dir"]) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running experiment: {config['experiment_name']}")
    print(f"Output directory: {output_dir}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Phase 1: Baseline
    baseline_metrics = run_baseline_evaluation(config, output_dir)

    # Phase 2: Client Training
    adapter_paths = run_client_training(config, output_dir)

    # Phase 3: Aggregation
    universal_path = run_aggregation(config, adapter_paths, output_dir)

    # Phase 4: Universal Evaluation
    universal_metrics = run_universal_evaluation(config, universal_path, output_dir)

    # Phase 5: Privacy Analysis
    privacy_metrics = run_privacy_analysis(config, adapter_paths, output_dir)

    # Summary
    print_summary(baseline_metrics, universal_metrics, privacy_metrics)

    print(f"\nExperiment complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FedLoRA-KD experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "baseline", "train", "aggregate", "evaluate", "privacy"],
                        help="Run specific phase only")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.phase == "all":
        main(config)
    else:
        set_seed(config["seed"])
        output_dir = Path(config["logging"]["output_dir"]) / config["experiment_name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.phase == "baseline":
            run_baseline_evaluation(config, output_dir)
        elif args.phase == "train":
            run_client_training(config, output_dir)
        # Add other phases as needed
