"""Model loading and LoRA setup for Llama 3.2."""

import os
import torch
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import login as hf_login


def setup_hf_auth() -> None:
    """Setup HuggingFace authentication from environment."""
    token = os.environ.get("HF_TOKEN")
    if token:
        hf_login(token=token, add_to_git_credential=False)
        print("HuggingFace: Authenticated via HF_TOKEN")
    else:
        print("HuggingFace: No HF_TOKEN found, using cached credentials")


def print_gpu_info() -> None:
    """Print GPU information."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n{'='*50}")
        print(f"GPU Connected: Yes")
        print(f"GPU Count: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  [{i}] {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"{'='*50}\n")
    else:
        print("\n" + "="*50)
        print("GPU Connected: No (using CPU)")
        print("="*50 + "\n")


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
    # Print GPU info
    print_gpu_info()

    # Setup HuggingFace auth
    setup_hf_auth()

    torch_dtype = getattr(torch, dtype)
    print(f"Loading model: {model_name}")
    print(f"Dtype: {dtype}")

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

    print(f"Model loaded successfully on device: {model.device}")

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
