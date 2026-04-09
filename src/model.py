"""Model loading and LoRA setup for Llama 3.2."""

import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel


def get_default_lora_config() -> Dict[str, Any]:
    """Return default LoRA configuration."""
    return {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }


def load_base_model(
    model_name: str = "meta-llama/Llama-3.2-3B",
    dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    device_map: str = "auto"
) -> tuple:
    """Load base model and tokenizer.

    Args:
        model_name: HuggingFace model identifier
        dtype: Data type (bfloat16, float16, float32)
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        device_map: Device placement strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.config.use_cache = False  # Required for gradient checkpointing

    return model, tokenizer


def setup_lora(model, lora_config: Optional[Dict[str, Any]] = None) -> PeftModel:
    """Apply LoRA adapters to model.

    Args:
        model: Base model
        lora_config: LoRA configuration dict

    Returns:
        PEFT model with LoRA adapters
    """
    if lora_config is None:
        lora_config = get_default_lora_config()

    peft_config = LoraConfig(**lora_config)
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    return peft_model


def load_adapter(base_model, adapter_path: str) -> PeftModel:
    """Load a saved LoRA adapter.

    Args:
        base_model: Base model to attach adapter to
        adapter_path: Path to saved adapter

    Returns:
        Model with loaded adapter
    """
    return PeftModel.from_pretrained(base_model, adapter_path)


def save_adapter(peft_model: PeftModel, save_path: str) -> None:
    """Save LoRA adapter weights.

    Args:
        peft_model: PEFT model with adapters
        save_path: Directory to save adapter
    """
    peft_model.save_pretrained(save_path)


def get_adapter_state_dict(peft_model: PeftModel) -> Dict[str, torch.Tensor]:
    """Extract LoRA adapter weights as state dict.

    Args:
        peft_model: PEFT model with adapters

    Returns:
        State dict containing only LoRA weights
    """
    state_dict = {}
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            state_dict[name] = param.detach().clone()
    return state_dict


def set_adapter_state_dict(peft_model: PeftModel, state_dict: Dict[str, torch.Tensor]) -> None:
    """Load LoRA weights from state dict.

    Args:
        peft_model: PEFT model to update
        state_dict: LoRA weights to load
    """
    model_state = peft_model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
