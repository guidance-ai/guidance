import pytest

import guidance
from guidance import gen, models


def test_call_embeddings():
    """This tests calls embedded in strings."""
    model = models.Mock()

    @guidance(dedent=False)
    def bla(lm, bla):
        lm += bla + "ae" + gen(max_tokens=10)
        return lm

    @guidance(dedent=False)
    def ble(lm):
        lm += f"""
    ae galera! {bla("33")}
    let's do more stuff!!""" + gen(max_tokens=10)
        return lm

    assert "{{G|" not in str(model + ble())


@pytest.mark.xfail(
    reason="llguidance currently emits an additional empty capture group when no explicit stop is provided"
)
def test_model_set():
    model = models.Mock()
    model = model.set("num", "4")
    assert "num" in model
    assert model["num"] == "4"
    assert model.log_prob("num") is not None

    model = model.set("list_num", ["1", "2"])
    assert "list_num" in model
    assert model["list_num"] == ["1", "2"]
    assert model.log_prob("list_num") is not None

    model += gen("list_num", max_tokens=10, list_append=True)
    assert len(model["list_num"]) == 3


def test_trace():
    from guidance import gen, models, system, user

    m0 = models.Mock()

    with system():
        m1 = m0 + "You are responsible for autocompleting a sentence."
    with user():
        m2 = m1 + "Roses are red and " + gen(name="suffix", regex="[A-Za-z]{2,5}", max_tokens=5)

    assert m2["suffix"] is not None


def test_step_every_k_injection():
    import re

    lm = models.Mock(echo=False)

    calls = {"count": 0}

    def cb(ctx):
        calls["count"] += 1
        return {"injected_text": "[FIX]"}

    cfg = {
        "step_every_k": 4,
        "callback": cb,
    }
    lm = lm.with_step_config(cfg)

    lm = lm + gen(max_tokens=20, stop="\n", temperature=0.0)

    s = str(lm)
    # find all occurrences of [FIX] in s and their positions
    occurrences = [m.start() for m in re.finditer(r"\[FIX\]", s)]
    assert occurrences == [6, 18]
    assert calls["count"] == len(occurrences)


def test_step_stop_token_trigger_injection():
    lm = models.Mock(byte_patterns=[b"abc!\n"], echo=False)

    calls = {"count": 0}

    def cb(ctx):
        calls["count"] += 1
        return {"injected_text": "[FIX2]"}

    cfg = {
        "step_stop_tokens": {"ym"},
        "callback": cb,
    }
    lm = lm.with_step_config(cfg)

    lm = lm + gen(max_tokens=20, stop="\n", temperature=0.0)

    s = str(lm)
    assert "[FIX2]" in s and "ym" not in s
    assert calls["count"] == 1


@pytest.mark.parametrize("outcome", ["complete", "error", "close", "start_error"])
def test_stream_restores_event_queues(outcome, monkeypatch):
    import queue
    import threading

    from guidance.models._base._model import _event_queues

    outer_queue = queue.Queue()
    outer_queues = (outer_queue,)
    token = _event_queues.set(outer_queues)
    try:
        model = models.Mock(echo=False)
        failure = RuntimeError("stream worker failed")

        @guidance(dedent=False)
        def fail(lm):
            raise failure

        stream = model.stream() + (fail() if outcome == "error" else "hello")
        iterator = iter(stream)
        if outcome == "error":
            with pytest.raises(RuntimeError) as exc:
                list(iterator)
            assert exc.value is failure
        elif outcome == "start_error":

            def fail_start(self):
                raise failure

            monkeypatch.setattr(threading.Thread, "start", fail_start)
            with pytest.raises(RuntimeError) as exc:
                next(iterator)
            assert exc.value is failure
            monkeypatch.undo()
        elif outcome == "close":
            next(iterator)
            iterator.close()
        else:
            assert str(list(iterator)[-1]) == "hello"

        assert _event_queues.get() is outer_queues
        size_before = outer_queue.qsize()
        model + "unrelated"
        assert outer_queue.qsize() > size_before
    finally:
        _event_queues.reset(token)
