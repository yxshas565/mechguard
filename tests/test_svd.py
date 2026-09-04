import numpy as np


def test_svd_reconstructs_matrix():
    rng = np.random.default_rng(42)

    matrix = rng.normal(size=(20, 10))

    u, s, vh = np.linalg.svd(matrix, full_matrices=False)

    reconstructed = (u * s) @ vh

    assert reconstructed.shape == matrix.shape
    assert np.allclose(reconstructed, matrix, atol=1e-8)


def test_singular_values_are_sorted():
    rng = np.random.default_rng(42)

    matrix = rng.normal(size=(20, 10))

    _, singular_values, _ = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    assert np.all(np.diff(singular_values) <= 0)