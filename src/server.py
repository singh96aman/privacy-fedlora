"""Server-side federated learning logic."""

from typing import Dict, List, Optional
from pathlib import Path

import torch

from .model import load_base_model, setup_lora, load_adapter, save_adapter
from .aggregator import fedavg_lora, save_aggregated_adapter


class Server:
    """Federated learning server with adapter aggregation."""

    def __init__(self, config: Dict) -> None:
        """Initialize server.

        Args:
            config: Full experiment configuration
        """
        self.config = config
        self.base_model = None
        self.tokenizer = None
        self.universal_adapter = None
        self.round = 0
        self.client_weights: Dict[str, torch.Tensor] = {}

    def initialize_model(self) -> tuple:
        """Initialize base model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        model_config = self.config["model"]
        self.base_model, self.tokenizer = load_base_model(
            model_name=model_config["name"],
            dtype=model_config.get("dtype", "bfloat16"),
            gradient_checkpointing=model_config.get("gradient_checkpointing", True)
        )
        return self.base_model, self.tokenizer

    def receive_update(self, client_id: str, update: Dict[str, torch.Tensor]) -> None:
        """Receive adapter update from client.

        Args:
            client_id: Client identifier
            update: LoRA adapter state dict
        """
        self.client_weights[client_id] = update
        print(f"Received update from client {client_id}")

    def aggregate(
        self,
        updates: Optional[List[Dict[str, torch.Tensor]]] = None,
        weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using FedAvg.

        Args:
            updates: List of adapter state dicts (if None, uses stored updates)
            weights: Optional weights for each client

        Returns:
            Aggregated adapter state dict
        """
        if updates is None:
            updates = list(self.client_weights.values())

        if not updates:
            raise ValueError("No updates to aggregate")

        self.universal_adapter = fedavg_lora(updates, weights)
        self.round += 1

        print(f"Aggregated {len(updates)} client updates (round {self.round})")
        return self.universal_adapter

    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """Return current universal adapter.

        Returns:
            Universal adapter state dict
        """
        if self.universal_adapter is None:
            raise RuntimeError("No universal adapter yet - run aggregate first")
        return self.universal_adapter

    def save_universal_adapter(
        self,
        output_dir: str,
        reference_adapter_path: str
    ) -> str:
        """Save universal adapter to disk.

        Args:
            output_dir: Base output directory
            reference_adapter_path: Path to copy config from

        Returns:
            Path where adapter was saved
        """
        if self.universal_adapter is None:
            raise RuntimeError("No universal adapter to save")

        save_path = Path(output_dir) / "universal" / "aggregated_adapter"
        save_path.mkdir(parents=True, exist_ok=True)

        save_aggregated_adapter(
            self.universal_adapter,
            str(save_path),
            reference_adapter_path
        )

        print(f"Saved universal adapter to {save_path}")
        return str(save_path)

    def get_universal_model(self):
        """Get base model with universal adapter loaded.

        Returns:
            PeftModel with universal adapter
        """
        if self.base_model is None:
            self.initialize_model()

        if self.universal_adapter is None:
            raise RuntimeError("No universal adapter - run aggregate first")

        # Setup fresh LoRA and load aggregated weights
        lora_config = self.config.get("lora", {})
        peft_model = setup_lora(self.base_model, lora_config)

        from .model import set_adapter_state_dict
        set_adapter_state_dict(peft_model, self.universal_adapter)

        return peft_model

    def select_clients(self, num_clients: int) -> List[str]:
        """Select clients for current round.

        Args:
            num_clients: Number of clients to select

        Returns:
            List of client IDs
        """
        all_clients = list(self.config["clients"].keys())
        return all_clients[:num_clients]
