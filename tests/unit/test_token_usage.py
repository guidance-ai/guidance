from guidance._schema import TokenUsage


def _usage(**kwargs) -> TokenUsage:
    return TokenUsage(round_trips=1, **kwargs)


def test_ttft_keeps_the_first_measurement():
    """`ttft_ms`/`ttfm_ms` are "time to *first* token/mask", so accumulating must keep the earliest.

    The producers set them once, guarded with `if usage.ttft_ms == 0`, so 0.0 means "not measured
    in this chunk". `__add__` used to take `other`'s value unconditionally, which overwrote the
    session's first-token latency with the latest chunk's on every addition.
    """
    first = _usage(forward_passes=10, ttft_ms=120.0, ttfm_ms=8.0, total_latency_ms=500.0)
    second = _usage(forward_passes=5, ttft_ms=95.0, ttfm_ms=6.0, total_latency_ms=300.0)

    total = first + second

    assert total.ttft_ms == 120.0
    assert total.ttfm_ms == 8.0


def test_a_chunk_without_generated_tokens_does_not_erase_ttft():
    """A chunk that generated nothing reports 0.0 and must not wipe an accumulated measurement.

    This is reachable whenever a completion loop is entered but issues no token (fully
    fast-forwarded or stopped immediately): `total_latency_ms` still accrues while `ttft_ms`
    stays at its 0.0 default.
    """
    measured = _usage(forward_passes=10, ttft_ms=120.0, ttfm_ms=8.0, total_latency_ms=500.0)
    no_tokens = _usage(forward_passes=0, ttft_ms=0.0, ttfm_ms=0.0, total_latency_ms=40.0)

    total = measured + no_tokens

    assert total.ttft_ms == 120.0
    assert total.ttfm_ms == 8.0


def test_ttft_is_taken_from_the_other_side_when_unset():
    """Accumulating onto a fresh `TokenUsage()` still picks up the first real measurement."""
    measured = _usage(forward_passes=10, ttft_ms=120.0, ttfm_ms=8.0)

    assert (TokenUsage() + measured).ttft_ms == 120.0
    assert (TokenUsage() + measured).ttfm_ms == 8.0


def test_repeated_accumulation_keeps_the_first_and_sums_the_rest():
    """The shape `State.add_usage` uses: `self._token_usage += usage` in a loop."""
    chunks = [
        _usage(forward_passes=10, ttft_ms=120.0, ttfm_ms=8.0, total_latency_ms=500.0),
        _usage(forward_passes=5, ttft_ms=95.0, ttfm_ms=6.0, total_latency_ms=300.0),
        _usage(forward_passes=0, ttft_ms=0.0, ttfm_ms=0.0, total_latency_ms=40.0),
    ]

    total = TokenUsage()
    for chunk in chunks:
        total += chunk

    assert total.ttft_ms == 120.0
    assert total.ttfm_ms == 8.0
    # the additive fields must keep adding
    assert total.round_trips == 3
    assert total.forward_passes == 15
    assert total.total_latency_ms == 840.0
