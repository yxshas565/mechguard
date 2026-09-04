import numpy as np


def normalized_overlap(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    numerator = np.linalg.norm(a.T @ b, ord="fro") ** 2
    denominator = (
        np.linalg.norm(a.T @ a, ord="fro")
        * np.linalg.norm(b.T @ b, ord="fro")
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def test_identical_subspaces_have_high_overlap():
    rng = np.random.default_rng(42)

    a = rng.normal(size=(20, 4))

    overlap = normalized_overlap(a, a)

    assert overlap > 0.99


def test_overlap_is_bounded():
    rng = np.random.default_rng(42)

    a = rng.normal(size=(20, 4))
    b = rng.normal(size=(20, 4))

    overlap = normalized_overlap(a, b)

    assert 0 <= overlap <= 1 + 1e-10