import numpy as np
import torch

from study_a.monitor import (
    cosine_similarity,
    frobenius_distance,
    randomized_svd_top_k,
)


def test_randomized_svd_is_deterministic():
    rng = np.random.default_rng(42)

    matrix = torch.tensor(
        rng.normal(size=(30, 15)),
        dtype=torch.float32,
    )

    u1, s1, vt1 = randomized_svd_top_k(
        matrix,
        k=5,
        seed=42,
    )

    u2, s2, vt2 = randomized_svd_top_k(
        matrix,
        k=5,
        seed=42,
    )

    # Singular values should be identical.
    assert torch.allclose(
        s1,
        s2,
        atol=1e-6,
    )

    # Subspaces should also be identical.
    projection_1 = u1 @ u1.T
    projection_2 = u2 @ u2.T

    assert torch.allclose(
        projection_1,
        projection_2,
        atol=1e-5,
    )


def test_frobenius_distance_zero_for_identical_matrices():
    matrix = torch.eye(5)

    distance = frobenius_distance(
        matrix,
        matrix,
    )

    assert np.isclose(
        distance,
        0.0,
        atol=1e-8,
    )


def test_frobenius_distance_is_positive_for_different_matrices():
    a = torch.eye(5)
    b = torch.zeros(5, 5)

    distance = frobenius_distance(
        a,
        b,
    )

    assert distance > 0.0


def test_cosine_similarity_identical_vectors():
    vector = torch.tensor(
        [1.0, 2.0, 3.0]
    )

    similarity = cosine_similarity(
        vector,
        vector,
    )

    assert np.isclose(
        similarity,
        1.0,
        atol=1e-8,
    )


def test_cosine_similarity_orthogonal_vectors():
    a = torch.tensor(
        [1.0, 0.0]
    )

    b = torch.tensor(
        [0.0, 1.0]
    )

    similarity = cosine_similarity(
        a,
        b,
    )

    assert np.isclose(
        similarity,
        0.0,
        atol=1e-8,
    )