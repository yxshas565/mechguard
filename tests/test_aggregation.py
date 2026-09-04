import numpy as np

from study_b.aggregation import (
    peak_suspicion,
    deception_split,
    asymmetry_probe,
)


def test_peak_suspicion():
    scores = np.array([0.2, 0.8, 0.4, 0.7])

    assert peak_suspicion(scores) == 0.8


def test_deception_split():
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    assert np.isclose(
        deception_split(scores),
        0.6,
    )


def test_asymmetry_probe():
    scores = np.array([0.9, 0.8, 0.2, 0.1])

    assert np.isclose(
        asymmetry_probe(scores),
        0.7,
    )