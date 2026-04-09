"""Privacy attack implementations for leakage analysis."""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def membership_inference_attack(
    member_losses: List[float],
    non_member_losses: List[float],
    test_size: float = 0.3
) -> Dict[str, float]:
    """Run membership inference attack using loss values.

    Attack model: train classifier to distinguish members from non-members
    based on their loss values (members typically have lower loss).

    Args:
        member_losses: Loss values for training set members
        non_member_losses: Loss values for non-training samples
        test_size: Fraction for test split

    Returns:
        Attack metrics (accuracy, precision, recall, AUC)
    """
    # Create features (loss) and labels (1=member, 0=non-member)
    X_member = np.array(member_losses).reshape(-1, 1)
    X_non_member = np.array(non_member_losses).reshape(-1, 1)

    X = np.vstack([X_member, X_non_member])
    y = np.array([1] * len(member_losses) + [0] * len(non_member_losses))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Train attack model
    attack_model = LogisticRegression()
    attack_model.fit(X_train, y_train)

    # Evaluate
    y_pred = attack_model.predict(X_test)
    y_proba = attack_model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_proba),
        "num_members": len(member_losses),
        "num_non_members": len(non_member_losses),
        "baseline_accuracy": max(len(member_losses), len(non_member_losses)) / (len(member_losses) + len(non_member_losses))
    }


def domain_identification_attack(
    adapter_weights: Dict[str, np.ndarray],
    client_labels: List[str],
    test_adapters: Dict[str, np.ndarray] = None,
    test_labels: List[str] = None
) -> Dict[str, float]:
    """Run domain identification attack on adapter weights.

    Attack model: given adapter weights, predict which domain/client
    contributed to them.

    Args:
        adapter_weights: Dict mapping adapter name to flattened weight vector
        client_labels: Label for each adapter (domain name)
        test_adapters: Optional test set adapters
        test_labels: Optional test set labels

    Returns:
        Attack metrics
    """
    # Flatten adapter weights into feature vectors
    X = []
    y = []

    for name, weights in adapter_weights.items():
        if isinstance(weights, dict):
            # Flatten all LoRA weights into single vector
            flat = np.concatenate([w.cpu().numpy().flatten() for w in weights.values()])
        else:
            flat = weights.flatten()
        X.append(flat)

    X = np.array(X)
    y = np.array(client_labels)

    # If no test set provided, use cross-validation style split
    if test_adapters is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_test = []
        for name, weights in test_adapters.items():
            if isinstance(weights, dict):
                flat = np.concatenate([w.cpu().numpy().flatten() for w in weights.values()])
            else:
                flat = weights.flatten()
            X_test.append(flat)
        X_test = np.array(X_test)
        y_test = np.array(test_labels)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)

    unique_labels = list(set(client_labels))
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "num_domains": len(unique_labels),
        "domains": unique_labels,
        "random_baseline": 1.0 / len(unique_labels)
    }


def analyze_weight_statistics(
    adapter_weights: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """Compute statistics on adapter weights for analysis.

    Args:
        adapter_weights: Adapter state dict

    Returns:
        Statistics per layer
    """
    stats = {}

    for name, weights in adapter_weights.items():
        if hasattr(weights, 'cpu'):
            w = weights.cpu().numpy()
        else:
            w = weights

        stats[name] = {
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
            "l2_norm": float(np.linalg.norm(w)),
            "sparsity": float(np.mean(np.abs(w) < 1e-6))
        }

    return stats


def compute_adapter_similarity(
    adapter1: Dict[str, np.ndarray],
    adapter2: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Compute similarity metrics between two adapters.

    Args:
        adapter1: First adapter weights
        adapter2: Second adapter weights

    Returns:
        Similarity metrics
    """
    # Flatten both
    flat1 = np.concatenate([
        (w.cpu().numpy() if hasattr(w, 'cpu') else w).flatten()
        for w in adapter1.values()
    ])
    flat2 = np.concatenate([
        (w.cpu().numpy() if hasattr(w, 'cpu') else w).flatten()
        for w in adapter2.values()
    ])

    # Cosine similarity
    cosine = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))

    # L2 distance
    l2_dist = np.linalg.norm(flat1 - flat2)

    return {
        "cosine_similarity": float(cosine),
        "l2_distance": float(l2_dist),
        "relative_l2": float(l2_dist / (np.linalg.norm(flat1) + np.linalg.norm(flat2)))
    }
