import numpy as np
import torch

from scipy.linalg import subspace_angles

from study_a.monitor import (
    compute_principal_angles,
    compute_subspace_overlap,
)


def test_identical_subspaces_have_zero_angle():
    rng = np.random.default_rng(42)

    a = rng.normal(size=(20, 4))

    angles = subspace_angles(a, a)

    assert np.max(angles) < 1e-10


def test_orthogonal_subspaces_have_90_degree_angles():
    # Two genuinely orthogonal 2D subspaces in R^4.
    a = np.eye(4)[:, :2]
    b = np.eye(4)[:, 2:]

    angles = compute_principal_angles(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
    )

    assert np.allclose(
        angles,
        90.0,
        atol=1e-5,
    )


def test_identical_subspaces_have_high_overlap():
    rng = np.random.default_rng(42)

    a = rng.normal(size=(20, 4))

    overlap = compute_subspace_overlap(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(a, dtype=torch.float32),
    )

    assert overlap > 0.99


def test_orthogonal_subspaces_have_low_overlap():
    a = np.eye(4)[:, :2]
    b = np.eye(4)[:, 2:]

    overlap = compute_subspace_overlap(
        torch.tensor(a, dtype=torch.float32),
        torch.tensor(b, dtype=torch.float32),
    )

    assert overlap < 1e-5