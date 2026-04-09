"""Training utilities for LoRA fine-tuning."""

import torch
from typing import Dict, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from peft import PeftModel


def train_lora(
    model: PeftModel,
    train_dataloader: DataLoader,
    config: Dict,
    eval_dataloader: Optional[DataLoader] = None
) -> Dict[str, float]:
    """Train LoRA adapter on client data.

    Args:
        model: PEFT model with LoRA adapters
        train_dataloader: Training data
        config: Training configuration
        eval_dataloader: Optional validation data

    Returns:
        Training metrics dict
    """
    training_config = config["training"]

    epochs = training_config.get("local_epochs", 3)
    lr = training_config.get("learning_rate", 2e-4)
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 8)

    optimizer = AdamW(model.parameters(), lr=lr)

    total_steps = len(train_dataloader) * epochs // grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    model.train()
    global_step = 0
    total_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(progress):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            progress.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        total_loss += avg_epoch_loss
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    metrics = {
        "train_loss": total_loss / epochs,
        "epochs": epochs,
        "global_steps": global_step
    }

    # Evaluate if validation data provided
    if eval_dataloader is not None:
        eval_loss = evaluate_loss(model, eval_dataloader)
        metrics["eval_loss"] = eval_loss

    return metrics


def evaluate_loss(model: PeftModel, dataloader: DataLoader) -> float:
    """Compute average loss on dataset.

    Args:
        model: Model to evaluate
        dataloader: Evaluation data

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    model.train()
    return total_loss / len(dataloader)
