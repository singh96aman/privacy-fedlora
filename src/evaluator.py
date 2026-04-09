"""Evaluation utilities for QA tasks."""

import re
import string
from typing import Dict, List, Tuple
from collections import Counter

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from peft import PeftModel


def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation.

    Args:
        s: Raw answer string

    Returns:
        Normalized answer
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def generate_answer(
    model,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 50
) -> str:
    """Generate answer from model.

    Args:
        model: Model to use for generation
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated answer string
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    generated = outputs[0][input_length:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)

    # Clean up answer - take first sentence/line
    answer = answer.split("\n")[0].strip()
    return answer


def evaluate_qa(
    model,
    tokenizer: PreTrainedTokenizer,
    eval_examples: List[Dict],
    max_samples: int = 500
) -> Dict[str, float]:
    """Evaluate model on QA examples.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_examples: List of {"prompt": str, "answer": str}
        max_samples: Maximum samples to evaluate

    Returns:
        Dict with f1 and exact_match scores
    """
    f1_scores = []
    em_scores = []

    samples = eval_examples[:max_samples]

    for example in tqdm(samples, desc="Evaluating QA"):
        prompt = example["prompt"]
        ground_truth = example["answer"]

        prediction = generate_answer(model, tokenizer, prompt)

        f1 = compute_f1(prediction, ground_truth)
        em = compute_exact_match(prediction, ground_truth)

        f1_scores.append(f1)
        em_scores.append(em)

    return {
        "f1": sum(f1_scores) / len(f1_scores),
        "exact_match": sum(em_scores) / len(em_scores),
        "num_samples": len(samples)
    }


def get_loss_distribution(
    model,
    dataloader,
    device: str = "cuda"
) -> List[float]:
    """Get per-sample loss distribution.

    Args:
        model: Model to evaluate
        dataloader: Data to compute loss on
        device: Device to use

    Returns:
        List of per-sample losses
    """
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing losses"):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Compute per-sample loss
            outputs = model(**batch)
            logits = outputs.logits

            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Average loss per sample
            loss = loss.view(shift_labels.size(0), -1).mean(dim=1)
            losses.extend(loss.cpu().tolist())

    return losses
