"""Knowledge Distillation trainer for dual-teacher learning."""

import torch
import torch.nn.functional as F
from typing import Dict, Optional
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from peft import PeftModel
import math


def compute_teacher_confidence(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0
) -> torch.Tensor:
    """Compute teacher confidence based on entropy of predictions.

    Args:
        teacher_logits: Logits from teacher model [batch, seq_len, vocab]
        labels: Ground truth labels [batch, seq_len], -100 for ignored positions
        temperature: Temperature for softmax

    Returns:
        Per-sample confidence scores [batch], in range [0, 1]
    """
    # Get teacher probabilities
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute entropy per token: -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    log_probs = torch.log(teacher_probs + 1e-10)
    entropy = -torch.sum(teacher_probs * log_probs, dim=-1)  # [batch, seq_len]

    # Mask out ignored positions (where labels == -100)
    mask = (labels != -100).float()

    # Average entropy over valid positions
    entropy_sum = (entropy * mask).sum(dim=-1)
    masked_entropy = entropy_sum / (mask.sum(dim=-1) + 1e-10)

    # Normalize by max possible entropy (log of vocab size)
    vocab_size = teacher_logits.size(-1)
    max_entropy = math.log(vocab_size)
    normalized_entropy = masked_entropy / max_entropy

    # Confidence = 1 - normalized_entropy
    confidence = 1.0 - normalized_entropy.clamp(0, 1)

    return confidence


def compute_confidence_weighted_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute confidence-weighted knowledge distillation loss.

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        labels: Ground truth labels
        temperature: Softmax temperature for distillation
        alpha: Base weight for distillation loss (scaled by confidence)

    Returns:
        Tuple of (combined loss, mean confidence for logging)
    """
    # Compute per-sample teacher confidence
    confidence = compute_teacher_confidence(teacher_logits, labels, temperature)

    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence loss per sample (not reduced)
    kd_loss_per_token = F.kl_div(
        soft_student, soft_targets, reduction="none"
    ).sum(dim=-1)

    # Mask and average over sequence
    mask = (labels != -100).float()
    kd_sum = (kd_loss_per_token * mask).sum(dim=-1)
    kd_loss_per_sample = kd_sum / (mask.sum(dim=-1) + 1e-10)
    kd_loss_per_sample = kd_loss_per_sample * (temperature ** 2)

    # Weight by confidence and average over batch
    weighted_kd_loss = (confidence * kd_loss_per_sample).mean()

    # Cross entropy loss (hard targets)
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    # Combined loss: CE + confidence-weighted KD
    combined_loss = (1 - alpha) * ce_loss + alpha * weighted_kd_loss

    return combined_loss, confidence.mean()


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5
) -> torch.Tensor:
    """Compute knowledge distillation loss.

    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        labels: Ground truth labels
        temperature: Softmax temperature for distillation
        alpha: Weight for distillation loss (1-alpha for CE loss)

    Returns:
        Combined KD loss
    """
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence loss (distillation)
    kd_loss = F.kl_div(soft_student, soft_targets, reduction="batchmean") * (temperature ** 2)

    # Cross entropy loss (hard targets)
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    return alpha * kd_loss + (1 - alpha) * ce_loss


def compute_dual_teacher_loss(
    student_logits: torch.Tensor,
    base_teacher_logits: torch.Tensor,
    universal_teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha_base: float = 0.25,
    alpha_universal: float = 0.25
) -> torch.Tensor:
    """Compute dual-teacher knowledge distillation loss.

    Combines:
    - Hard target CE loss (ground truth)
    - Soft target KD from base model (preserve general knowledge)
    - Soft target KD from universal teacher (domain knowledge)

    Args:
        student_logits: Logits from student model
        base_teacher_logits: Logits from frozen base model
        universal_teacher_logits: Logits from universal teacher
        labels: Ground truth labels
        temperature: Softmax temperature
        alpha_base: Weight for base model distillation
        alpha_universal: Weight for universal teacher distillation

    Returns:
        Combined loss
    """
    alpha_ce = 1.0 - alpha_base - alpha_universal

    # Soft targets from both teachers
    base_soft = F.softmax(base_teacher_logits / temperature, dim=-1)
    universal_soft = F.softmax(universal_teacher_logits / temperature, dim=-1)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)

    # KD losses
    kd_base = F.kl_div(student_log_soft, base_soft, reduction="batchmean") * (temperature ** 2)
    kd_universal = F.kl_div(student_log_soft, universal_soft, reduction="batchmean") * (temperature ** 2)

    # CE loss
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    return alpha_ce * ce_loss + alpha_base * kd_base + alpha_universal * kd_universal


def train_with_kd(
    student_model: PeftModel,
    teacher_model,
    train_dataloader: DataLoader,
    config: Dict,
    eval_dataloader: Optional[DataLoader] = None
) -> Dict[str, float]:
    """Train student model with single teacher KD.

    Args:
        student_model: Student PEFT model to train
        teacher_model: Teacher model (frozen)
        train_dataloader: Training data
        config: Training configuration
        eval_dataloader: Optional validation data

    Returns:
        Training metrics
    """
    training_config = config["training"]
    kd_config = config.get("knowledge_distillation", {})

    epochs = training_config.get("local_epochs", 3)
    lr = training_config.get("learning_rate", 2e-4)
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 8)
    temperature = kd_config.get("temperature", 2.0)
    alpha = kd_config.get("alpha", 0.5)

    optimizer = AdamW(student_model.parameters(), lr=lr)

    total_steps = len(train_dataloader) * epochs // grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    student_model.train()
    teacher_model.eval()

    global_step = 0
    total_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress = tqdm(train_dataloader, desc=f"KD Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(progress):
            batch = {k: v.to(student_model.device) for k, v in batch.items()}

            # Get teacher logits (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                teacher_logits = teacher_outputs.logits

            # Get student logits
            student_outputs = student_model(**batch)
            student_logits = student_outputs.logits

            # Compute KD loss
            loss = compute_kd_loss(
                student_logits,
                teacher_logits,
                batch["labels"],
                temperature=temperature,
                alpha=alpha
            )

            loss = loss / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            progress.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        total_loss += avg_epoch_loss
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    return {
        "train_loss": total_loss / epochs,
        "epochs": epochs,
        "global_steps": global_step
    }


def train_with_confidence_weighted_kd(
    student_model: PeftModel,
    teacher_model,
    train_dataloader: DataLoader,
    config: Dict,
    eval_dataloader: Optional[DataLoader] = None
) -> Dict[str, float]:
    """Train student with confidence-weighted KD from single teacher.

    The KD loss is weighted by teacher confidence (1 - normalized entropy).
    When teacher is uncertain, we rely more on ground truth labels.

    Args:
        student_model: Student PEFT model to train
        teacher_model: Teacher model (frozen)
        train_dataloader: Training data
        config: Training configuration
        eval_dataloader: Optional validation data

    Returns:
        Training metrics including average confidence
    """
    training_config = config["training"]
    kd_config = config.get("knowledge_distillation", {})

    epochs = training_config.get("local_epochs", 3)
    lr = training_config.get("learning_rate", 2e-4)
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 8)
    temperature = kd_config.get("temperature", 2.0)
    alpha = kd_config.get("alpha", 0.5)

    optimizer = AdamW(student_model.parameters(), lr=lr)

    total_steps = len(train_dataloader) * epochs // grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    student_model.train()
    teacher_model.eval()

    global_step = 0
    total_loss = 0.0
    total_confidence = 0.0
    num_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_confidence = 0.0
        desc = f"Confidence-KD Epoch {epoch + 1}/{epochs}"
        progress = tqdm(train_dataloader, desc=desc)

        for step, batch in enumerate(progress):
            batch = {k: v.to(student_model.device) for k, v in batch.items()}

            # Get teacher logits (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                teacher_logits = teacher_outputs.logits

            # Get student logits
            student_outputs = student_model(**batch)
            student_logits = student_outputs.logits

            # Compute confidence-weighted KD loss
            loss, confidence = compute_confidence_weighted_kd_loss(
                student_logits,
                teacher_logits,
                batch["labels"],
                temperature=temperature,
                alpha=alpha
            )

            loss = loss / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps
            epoch_confidence += confidence.item()
            num_batches += 1

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            progress.set_postfix({
                "loss": f"{loss.item() * grad_accum_steps:.4f}",
                "conf": f"{confidence.item():.3f}"
            })

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_epoch_conf = epoch_confidence / len(train_dataloader)
        total_loss += avg_epoch_loss
        total_confidence += avg_epoch_conf
        print(f"Epoch {epoch + 1} - loss: {avg_epoch_loss:.4f}, "
              f"avg confidence: {avg_epoch_conf:.3f}")

    return {
        "train_loss": total_loss / epochs,
        "avg_confidence": total_confidence / epochs,
        "epochs": epochs,
        "global_steps": global_step
    }


def train_with_dual_teacher_kd(
    student_model: PeftModel,
    base_teacher_model,
    universal_teacher_model,
    train_dataloader: DataLoader,
    config: Dict,
    eval_dataloader: Optional[DataLoader] = None
) -> Dict[str, float]:
    """Train student model with dual-teacher KD.

    Args:
        student_model: Student PEFT model to train
        base_teacher_model: Frozen base model (general knowledge)
        universal_teacher_model: Universal teacher (federated knowledge)
        train_dataloader: Training data
        config: Training configuration
        eval_dataloader: Optional validation data

    Returns:
        Training metrics
    """
    training_config = config["training"]
    kd_config = config.get("knowledge_distillation", {})

    epochs = training_config.get("local_epochs", 3)
    lr = training_config.get("learning_rate", 2e-4)
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 8)
    temperature = kd_config.get("temperature", 2.0)
    alpha_base = kd_config.get("alpha_base", 0.25)
    alpha_universal = kd_config.get("alpha_universal", 0.25)

    optimizer = AdamW(student_model.parameters(), lr=lr)

    total_steps = len(train_dataloader) * epochs // grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    student_model.train()
    base_teacher_model.eval()
    universal_teacher_model.eval()

    global_step = 0
    total_loss = 0.0

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress = tqdm(train_dataloader, desc=f"Dual-KD Epoch {epoch + 1}/{epochs}")

        for step, batch in enumerate(progress):
            batch = {k: v.to(student_model.device) for k, v in batch.items()}

            # Get teacher logits (no grad)
            with torch.no_grad():
                base_outputs = base_teacher_model(**batch)
                universal_outputs = universal_teacher_model(**batch)
                base_logits = base_outputs.logits
                universal_logits = universal_outputs.logits

            # Get student logits
            student_outputs = student_model(**batch)
            student_logits = student_outputs.logits

            # Compute dual-teacher KD loss
            loss = compute_dual_teacher_loss(
                student_logits,
                base_logits,
                universal_logits,
                batch["labels"],
                temperature=temperature,
                alpha_base=alpha_base,
                alpha_universal=alpha_universal
            )

            loss = loss / grad_accum_steps
            loss.backward()

            epoch_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            progress.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        total_loss += avg_epoch_loss
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

    return {
        "train_loss": total_loss / epochs,
        "epochs": epochs,
        "global_steps": global_step,
        "alpha_base": alpha_base,
        "alpha_universal": alpha_universal
    }
