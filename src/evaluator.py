"""Evaluation utilities for QA tasks."""

import re
import string
import math
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
import numpy as np
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


def compute_contains(prediction: str, ground_truth: str) -> float:
    """Check if ground truth is contained in prediction.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1.0 if ground truth is substring of prediction, 0.0 otherwise
    """
    return float(normalize_answer(ground_truth) in normalize_answer(prediction))


def compute_bleu(prediction: str, ground_truth: str) -> float:
    """Compute BLEU score (simplified unigram/bigram).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        BLEU score
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # Unigram precision
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    clipped = sum((pred_counter & ref_counter).values())
    precision = clipped / len(pred_tokens) if pred_tokens else 0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if pred_tokens else 0

    return bp * precision


def compute_rouge_l(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L score (longest common subsequence).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        ROUGE-L F1 score
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # LCS length
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0.0

    precision = lcs_length / m
    recall = lcs_length / n
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_perplexity(
    model,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: str = "cuda"
) -> float:
    """Compute perplexity for a given text.

    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Text to compute perplexity for
        device: Device to use

    Returns:
        Perplexity value
    """
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()

    return math.exp(loss)


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    device: str = "cuda"
) -> Dict[str, float]:
    """Compute BERTScore for predictions.

    Args:
        predictions: List of predicted answers
        references: List of reference answers
        device: Device to use

    Returns:
        Dict with precision, recall, f1
    """
    try:
        from bert_score import score as bert_score
        P, R, F1 = bert_score(
            predictions, references,
            lang="en",
            device=device,
            verbose=False
        )
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item()
        }
    except ImportError:
        # bert-score not installed, return zeros
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0
        }


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
    max_samples: int = 500,
    compute_all_metrics: bool = False
) -> Dict[str, float]:
    """Evaluate model on QA examples.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        eval_examples: List of {"prompt": str, "answer": str}
        max_samples: Maximum samples to evaluate
        compute_all_metrics: If True, compute all metrics (slower)

    Returns:
        Dict with evaluation metrics
    """
    f1_scores = []
    em_scores = []
    contains_scores = []
    bleu_scores = []
    rouge_scores = []
    ppl_scores = []
    predictions = []
    references = []

    samples = eval_examples[:max_samples]

    for example in tqdm(samples, desc="Evaluating QA"):
        prompt = example["prompt"]
        ground_truth = example["answer"]

        prediction = generate_answer(model, tokenizer, prompt)

        # Core metrics
        f1_scores.append(compute_f1(prediction, ground_truth))
        em_scores.append(compute_exact_match(prediction, ground_truth))

        if compute_all_metrics:
            contains_scores.append(compute_contains(prediction, ground_truth))
            bleu_scores.append(compute_bleu(prediction, ground_truth))
            rouge_scores.append(compute_rouge_l(prediction, ground_truth))

            # PPL on full response
            full_text = f"{prompt} {prediction}"
            ppl_scores.append(compute_perplexity(model, tokenizer, full_text))

            predictions.append(prediction)
            references.append(ground_truth)

    results = {
        "f1": np.mean(f1_scores),
        "exact_match": np.mean(em_scores),
        "num_samples": len(samples)
    }

    if compute_all_metrics:
        results.update({
            "contains": np.mean(contains_scores),
            "bleu": np.mean(bleu_scores),
            "rouge_l": np.mean(rouge_scores),
            "perplexity": np.mean(ppl_scores)
        })

        # BERTScore (batch computation)
        bertscore_results = compute_bertscore(predictions, references)
        results.update(bertscore_results)

    return results


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
