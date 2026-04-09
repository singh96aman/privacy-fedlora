"""Tests for privacy mechanisms and attacks."""

import pytest
import numpy as np
import torch


class TestMembershipInferenceAttack:
    """Tests for membership inference attack."""

    def test_perfect_separation(self):
        """Attack should achieve high accuracy with separable losses."""
        from src.attacks import membership_inference_attack

        # Members have lower loss, non-members have higher
        member_losses = np.random.normal(1.0, 0.1, 100).tolist()
        non_member_losses = np.random.normal(3.0, 0.1, 100).tolist()

        results = membership_inference_attack(member_losses, non_member_losses)

        assert results["accuracy"] > 0.9
        assert results["auc"] > 0.95

    def test_random_baseline(self):
        """Attack should be near random with overlapping distributions."""
        from src.attacks import membership_inference_attack

        # Same distribution for both
        member_losses = np.random.normal(2.0, 1.0, 100).tolist()
        non_member_losses = np.random.normal(2.0, 1.0, 100).tolist()

        results = membership_inference_attack(member_losses, non_member_losses)

        # Should be close to random (0.5)
        assert 0.4 < results["accuracy"] < 0.6


class TestDomainIdentificationAttack:
    """Tests for domain identification attack."""

    def test_distinct_domains(self):
        """Attack should identify clearly distinct domain signatures."""
        from src.attacks import domain_identification_attack

        # Create distinct adapter weight patterns
        adapters = {
            "c1_run1": {"layer": np.ones(100) * 1.0},
            "c1_run2": {"layer": np.ones(100) * 1.1},
            "c2_run1": {"layer": np.ones(100) * 5.0},
            "c2_run2": {"layer": np.ones(100) * 5.1},
        }
        labels = ["c1", "c1", "c2", "c2"]

        results = domain_identification_attack(adapters, labels)

        assert results["accuracy"] > 0.5  # Better than random


class TestFedAvg:
    """Tests for federated averaging."""

    def test_simple_average(self):
        """FedAvg should compute simple average with equal weights."""
        from src.aggregator import fedavg_lora

        adapter1 = {"layer.weight": torch.tensor([1.0, 2.0, 3.0])}
        adapter2 = {"layer.weight": torch.tensor([3.0, 4.0, 5.0])}

        result = fedavg_lora([adapter1, adapter2])

        expected = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(result["layer.weight"], expected)

    def test_weighted_average(self):
        """FedAvg should respect client weights."""
        from src.aggregator import fedavg_lora

        adapter1 = {"layer.weight": torch.tensor([0.0, 0.0])}
        adapter2 = {"layer.weight": torch.tensor([10.0, 10.0])}

        # Weight adapter2 3x more
        result = fedavg_lora([adapter1, adapter2], weights=[1.0, 3.0])

        expected = torch.tensor([7.5, 7.5])  # (0*0.25 + 10*0.75)
        assert torch.allclose(result["layer.weight"], expected)


class TestQAMetrics:
    """Tests for QA evaluation metrics."""

    def test_exact_match(self):
        """Exact match should handle normalization."""
        from src.evaluator import compute_exact_match

        assert compute_exact_match("The answer", "the answer") == 1.0
        assert compute_exact_match("answer", "the answer") == 1.0
        assert compute_exact_match("wrong", "answer") == 0.0

    def test_f1_score(self):
        """F1 should compute token overlap."""
        from src.evaluator import compute_f1

        # Perfect match
        assert compute_f1("the cat", "the cat") == 1.0

        # Partial overlap
        f1 = compute_f1("the big cat", "the small cat")
        assert 0 < f1 < 1

        # No overlap
        assert compute_f1("apple", "orange") == 0.0


class TestAdapterSimilarity:
    """Tests for adapter similarity computation."""

    def test_identical_adapters(self):
        """Identical adapters should have similarity 1.0."""
        from src.attacks import compute_adapter_similarity

        adapter = {"layer": torch.tensor([1.0, 2.0, 3.0])}

        result = compute_adapter_similarity(adapter, adapter)

        assert result["cosine_similarity"] == pytest.approx(1.0)
        assert result["l2_distance"] == pytest.approx(0.0)

    def test_orthogonal_adapters(self):
        """Orthogonal adapters should have similarity 0."""
        from src.attacks import compute_adapter_similarity

        adapter1 = {"layer": torch.tensor([1.0, 0.0])}
        adapter2 = {"layer": torch.tensor([0.0, 1.0])}

        result = compute_adapter_similarity(adapter1, adapter2)

        assert result["cosine_similarity"] == pytest.approx(0.0, abs=1e-6)
