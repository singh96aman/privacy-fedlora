"""Client-side federated learning logic."""

from typing import Dict, Optional
from pathlib import Path

import torch
from peft import PeftModel

from .model import load_base_model, setup_lora, get_adapter_state_dict, save_adapter
from .data import get_client_data, create_dataloader
from .trainer import train_lora


class Client:
    """Federated learning client with LoRA training."""

    def __init__(
        self,
        client_id: str,
        config: Dict,
        tokenizer=None,
        base_model=None
    ) -> None:
        """Initialize client.

        Args:
            client_id: Client identifier (c1, c2, etc.)
            config: Full experiment configuration
            tokenizer: Shared tokenizer (optional, will load if None)
            base_model: Shared base model (optional, will load if None)
        """
        self.client_id = client_id
        self.config = config
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.peft_model: Optional[PeftModel] = None
        self.train_dataset = None
        self.eval_dataset = None
        self.training_metrics = None

    def setup(self) -> None:
        """Setup model and data for training."""
        # Load base model if not provided
        if self.base_model is None or self.tokenizer is None:
            model_config = self.config["model"]
            self.base_model, self.tokenizer = load_base_model(
                model_name=model_config["name"],
                dtype=model_config.get("dtype", "bfloat16"),
                gradient_checkpointing=model_config.get("gradient_checkpointing", True)
            )

        # Setup LoRA
        lora_config = self.config.get("lora", {})
        self.peft_model = setup_lora(self.base_model, lora_config)

        # Load data
        self.train_dataset, self.eval_dataset = get_client_data(
            self.client_id,
            self.config,
            self.tokenizer
        )
        print(f"Client {self.client_id}: {len(self.train_dataset)} train, {len(self.eval_dataset)} eval samples")

    def train(self) -> Dict:
        """Run local training and return metrics.

        Returns:
            Training metrics dict
        """
        if self.peft_model is None:
            self.setup()

        training_config = self.config["training"]
        batch_size = training_config.get("batch_size", 4)

        train_loader = create_dataloader(self.train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = create_dataloader(self.eval_dataset, batch_size=batch_size, shuffle=False)

        self.training_metrics = train_lora(
            self.peft_model,
            train_loader,
            self.config,
            eval_dataloader=eval_loader
        )

        return self.training_metrics

    def get_update(self) -> Dict[str, torch.Tensor]:
        """Return LoRA adapter weights.

        Returns:
            State dict with LoRA parameters
        """
        if self.peft_model is None:
            raise RuntimeError("Model not trained yet")

        return get_adapter_state_dict(self.peft_model)

    def save(self, output_dir: str) -> str:
        """Save adapter to disk.

        Args:
            output_dir: Base output directory

        Returns:
            Path where adapter was saved
        """
        if self.peft_model is None:
            raise RuntimeError("Model not trained yet")

        save_path = Path(output_dir) / f"client_{self.client_id}" / "lora_adapter"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        save_adapter(self.peft_model, str(save_path))
        print(f"Saved adapter for client {self.client_id} to {save_path}")

        return str(save_path)

    def set_model(self, model_state: Dict[str, torch.Tensor]) -> None:
        """Load adapter weights into model.

        Args:
            model_state: LoRA adapter state dict
        """
        if self.peft_model is None:
            self.setup()

        from .model import set_adapter_state_dict
        set_adapter_state_dict(self.peft_model, model_state)
