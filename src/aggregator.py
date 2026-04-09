"""Federated aggregation utilities for LoRA adapters."""

import torch
from typing import Dict, List
from pathlib import Path
from peft import PeftModel


def fedavg_lora(
    adapter_state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float] = None
) -> Dict[str, torch.Tensor]:
    """Aggregate LoRA adapters using FedAvg.

    Args:
        adapter_state_dicts: List of adapter state dicts from clients
        weights: Optional weights for each client (defaults to equal)

    Returns:
        Aggregated adapter state dict
    """
    if not adapter_state_dicts:
        raise ValueError("No adapters to aggregate")

    num_clients = len(adapter_state_dicts)

    if weights is None:
        weights = [1.0 / num_clients] * num_clients
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    # Initialize aggregated dict with zeros
    aggregated = {}
    first_dict = adapter_state_dicts[0]

    for key in first_dict:
        aggregated[key] = torch.zeros_like(first_dict[key])

    # Weighted sum
    for state_dict, weight in zip(adapter_state_dicts, weights):
        for key in state_dict:
            aggregated[key] += weight * state_dict[key]

    return aggregated


def load_adapter_weights(adapter_path: str) -> Dict[str, torch.Tensor]:
    """Load adapter weights from saved checkpoint.

    Args:
        adapter_path: Path to adapter directory

    Returns:
        State dict with LoRA weights
    """
    path = Path(adapter_path)
    adapter_file = path / "adapter_model.bin"

    if adapter_file.exists():
        return torch.load(adapter_file, map_location="cpu")

    # Try safetensors format
    safetensor_file = path / "adapter_model.safetensors"
    if safetensor_file.exists():
        from safetensors.torch import load_file
        return load_file(safetensor_file)

    raise FileNotFoundError(f"No adapter found at {adapter_path}")


def save_aggregated_adapter(
    aggregated_weights: Dict[str, torch.Tensor],
    output_path: str,
    base_adapter_path: str
) -> None:
    """Save aggregated adapter weights.

    Copies config from base adapter and saves new weights.

    Args:
        aggregated_weights: Aggregated LoRA weights
        output_path: Where to save
        base_adapter_path: Path to copy config from
    """
    import shutil
    from pathlib import Path

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    base_path = Path(base_adapter_path)

    # Copy config files
    for config_file in ["adapter_config.json", "README.md"]:
        src = base_path / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)

    # Save weights
    torch.save(aggregated_weights, output_path / "adapter_model.bin")


def aggregate_from_paths(
    adapter_paths: List[str],
    output_path: str,
    weights: List[float] = None
) -> Dict[str, torch.Tensor]:
    """Load adapters from paths, aggregate, and save.

    Args:
        adapter_paths: List of paths to adapter directories
        output_path: Where to save aggregated adapter
        weights: Optional client weights

    Returns:
        Aggregated weights dict
    """
    # Load all adapters
    adapter_dicts = []
    for path in adapter_paths:
        weights_dict = load_adapter_weights(path)
        adapter_dicts.append(weights_dict)
        print(f"Loaded adapter from {path}")

    # Aggregate
    aggregated = fedavg_lora(adapter_dicts, weights)
    print(f"Aggregated {len(adapter_dicts)} adapters")

    # Save
    save_aggregated_adapter(aggregated, output_path, adapter_paths[0])
    print(f"Saved aggregated adapter to {output_path}")

    return aggregated
