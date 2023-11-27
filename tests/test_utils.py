"""
Unit tests guidance._utils
"""
from typing import Tuple

import numpy as np
import pytest
import torch

from guidance import _utils


@pytest.fixture(scope="module")
def atol() -> float:
    return 1e-6


@pytest.mark.parametrize(
    "size_and_axis",
    [
        ((32_000,), -1),  # very next token logits
        ((10, 32_000), -1),  # many token's next-token logits
        ((4, 10, 32_000), -1),  # batch of texts
    ],
)
class TestLogitsTransforms:
    def test_log_softmax(self, size_and_axis, atol: float):
        size, axis = size_and_axis
        logits: np.ndarray = -np.random.uniform(low=0, high=60, size=size)
        log_probs = _utils.log_softmax(logits, axis=axis)
        log_probs_correct = torch.tensor(logits).log_softmax(dim=axis).numpy()
        assert np.allclose(log_probs, log_probs_correct, atol=atol)

    def test_softmax(self, size_and_axis, atol: float):
        size, axis = size_and_axis
        logits: np.ndarray = -np.random.uniform(low=0, high=60, size=size)
        probs = _utils.softmax(logits, axis=axis)
        probs_correct = torch.tensor(logits).softmax(dim=axis).numpy()
        assert np.allclose(probs, probs_correct, atol=atol)
