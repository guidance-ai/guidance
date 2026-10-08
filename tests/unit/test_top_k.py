import numpy as np
import pytest

from guidance._utils import apply_top_k_and_top_p_filter, apply_top_k_only
from guidance.models._engine._engine import Engine, get_top_k
from guidance.models._mock import MockTokenizer


@pytest.mark.parametrize("k", [3, 4, 50])
def test_top_k_at_least_vocabulary_size_keeps_all_logits(k):
    logits = np.array([-2.0, 3.0, 1.0])
    expected = logits.copy()

    assert apply_top_k_only(logits, k) is logits
    np.testing.assert_array_equal(logits, expected)
    np.testing.assert_array_equal(apply_top_k_and_top_p_filter(expected.copy(), {"top_k": k}), expected)


@pytest.mark.parametrize("k", [-1, 0, 1, 2])
def test_top_k_filter_preserves_existing_cutoffs(k):
    logits = np.array([-2.0, 3.0, 1.0])
    expected = logits.copy()
    if k > 0:
        expected[np.argsort(expected)[:-k]] = -np.inf

    np.testing.assert_array_equal(apply_top_k_only(logits, k), expected)


@pytest.mark.parametrize("k", [3, 4, 50])
def test_top_k_at_least_vocabulary_size_still_applies_top_p(k):
    logits = np.array([-2.0, 3.0, 1.0])
    expected = apply_top_k_and_top_p_filter(logits.copy(), {"top_p": 0.8})

    np.testing.assert_array_equal(apply_top_k_and_top_p_filter(logits.copy(), {"top_k": k, "top_p": 0.8}), expected)


@pytest.mark.parametrize("k, expected", [(-1, []), (0, []), (1, [1]), (2, [1, 2]), (3, [1, 2, 0]), (4, [1, 2, 0])])
def test_get_top_k_returns_at_most_the_vocabulary_size(k, expected):
    probs = np.array([0.1, 0.6, 0.3])
    np.testing.assert_array_equal(get_top_k(probs, k), expected)


class _TestEngine(Engine):
    def get_logits(self, token_ids, include_all_uncached_tokens=False):
        raise AssertionError("This test supplies logits directly")


def test_engine_top_k_metadata_can_exceed_a_custom_vocabulary():
    engine = _TestEngine(MockTokenizer([b"<s>", b"a", b"b"]), enable_monitoring=False)
    token = engine.get_next_token_with_top_k(
        logits=np.array([-2.0, 3.0, 1.0]),
        logits_lat_ms=0.0,
        token_ids=[],
        mask=None,
        temperature=0.0,
        k=5,
        compute_unmasked_probs=True,
        sampling_params=None,
    )

    assert token.token_id == 1
    assert [item.token_id for item in token.top_k] == [1, 2, 0]
    assert [item.bytes for item in token.top_k] == [b"a", b"b", b"<s>"]


def test_engine_sampling_top_k_can_exceed_the_vocabulary_size():
    engine = _TestEngine(MockTokenizer([b"<s>", b"a", b"b"]), enable_monitoring=False)
    token = engine.get_next_token_with_top_k(
        logits=np.array([-2.0, 3.0, 1.0]),
        logits_lat_ms=0.0,
        token_ids=[],
        mask=bytes([0, 1, 0]),
        temperature=1.0,
        k=0,
        compute_unmasked_probs=False,
        sampling_params={"top_k": 50},
    )

    assert token.token_id == 1
    assert token.bytes == b"a"
