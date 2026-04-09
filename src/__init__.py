"""Privacy-preserving federated learning source code."""

from .model import (
    load_base_model,
    setup_lora,
    load_adapter,
    save_adapter,
    get_adapter_state_dict,
    set_adapter_state_dict,
    get_default_lora_config
)

from .data import (
    load_squad,
    load_natural_questions,
    format_qa_prompt,
    format_squad_example,
    format_nq_example,
    preprocess_dataset,
    create_dataloader,
    get_client_data
)

from .trainer import train_lora, evaluate_loss

from .evaluator import (
    compute_f1,
    compute_exact_match,
    evaluate_qa,
    generate_answer,
    get_loss_distribution
)

from .aggregator import (
    fedavg_lora,
    load_adapter_weights,
    save_aggregated_adapter,
    aggregate_from_paths
)

from .attacks import (
    membership_inference_attack,
    domain_identification_attack,
    analyze_weight_statistics,
    compute_adapter_similarity
)

from .client import Client
from .server import Server

__all__ = [
    # Model
    "load_base_model",
    "setup_lora",
    "load_adapter",
    "save_adapter",
    "get_adapter_state_dict",
    "set_adapter_state_dict",
    "get_default_lora_config",
    # Data
    "load_squad",
    "load_natural_questions",
    "format_qa_prompt",
    "format_squad_example",
    "format_nq_example",
    "preprocess_dataset",
    "create_dataloader",
    "get_client_data",
    # Training
    "train_lora",
    "evaluate_loss",
    # Evaluation
    "compute_f1",
    "compute_exact_match",
    "evaluate_qa",
    "generate_answer",
    "get_loss_distribution",
    # Aggregation
    "fedavg_lora",
    "load_adapter_weights",
    "save_aggregated_adapter",
    "aggregate_from_paths",
    # Attacks
    "membership_inference_attack",
    "domain_identification_attack",
    "analyze_weight_statistics",
    "compute_adapter_similarity",
    # Client/Server
    "Client",
    "Server",
]
