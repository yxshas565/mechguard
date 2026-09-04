import numpy as np
import torch

from study_a.monitor import (
    compute_commutator_defect,
)


def test_commutator_defect_is_zero_for_commuting_updates():
    theta_0 = torch.zeros(10)

    grad_a = torch.ones(10)
    grad_b = torch.ones(10)

    # If the gradients do not change after the other update,
    # order should not matter.
    grad_b_after_a = grad_b.clone()
    grad_a_after_b = grad_a.clone()

    defect = compute_commutator_defect(
        theta_0=theta_0,
        grad_a=grad_a,
        grad_b=grad_b,
        grad_b_after_a=grad_b_after_a,
        grad_a_after_b=grad_a_after_b,
        learning_rate=0.01,
    )

    assert np.isclose(
        defect,
        0.0,
        atol=1e-8,
    )


def test_commutator_defect_is_positive_for_order_dependent_updates():
    theta_0 = torch.zeros(10)

    grad_a = torch.ones(10)
    grad_b = torch.ones(10)

    # Construct an explicitly order-dependent case.
    grad_b_after_a = grad_b + 0.5
    grad_a_after_b = grad_a - 0.5

    defect = compute_commutator_defect(
        theta_0=theta_0,
        grad_a=grad_a,
        grad_b=grad_b,
        grad_b_after_a=grad_b_after_a,
        grad_a_after_b=grad_a_after_b,
        learning_rate=0.01,
    )

    assert defect > 0.0