import asyncio
import pytest
from guidance import guidance, models, select
from guidance._bridge import AwaitException

@guidance
def sync_func(lm: models.Model):
    lm += select(["a", "b"], name="choice")
    if lm.get("choice") == "a":
        lm += "lpha"
    else:
        lm += "eta"
    return lm

@guidance
async def async_func(lm: models.Model):
    lm += select(["a", "b"], name="choice")
    if (await lm.get_async("choice")) == "a":
        lm += "lpha"
    else:
        lm += "eta"
    return lm

@guidance
async def async_func_with_sync_accessor(lm: models.Model):
    lm += select(["a", "b"], name="choice")
    if lm.get("choice") == "a":
        lm += "lpha"
    else:
        lm += "eta"
    return lm

@guidance
def sync_func_calling_async_func(lm: models.Model):
    lm += async_func()
    lm += select(["a", "b"], name="choice")
    if lm.get("choice") == "a":
        lm += "lpha"
    else:
        lm += "eta"
    return lm

@guidance
async def async_func_calling_sync_func(lm: models.Model):
    lm += sync_func()
    lm += select(["a", "b"], name="choice")
    if (await lm.get_async("choice")) == "a":
        lm += "lpha"
    else:
        lm += "eta"
    return lm

@guidance
def sync_then_async(lm: models.Model):
    lm += sync_func()
    lm += async_func()
    return lm

@guidance
def async_then_sync(lm: models.Model):
    lm += async_func()
    lm += sync_func()
    return lm

def run(gfunc, sync: bool) -> str:
    lm = models.Mock()
    lm += gfunc()
    if sync:
        s = str(lm)
    else:
        s = asyncio.run(lm.to_string_async())
    return s

@pytest.mark.parametrize(
    "sync",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "gfunc, expected",
    [
        (sync_func, {"alpha", "beta"}),
        (async_func, {"alpha", "beta"}),
        (sync_then_async, {"alphabeta", "betaalpha", "alphaalpha", "betabeta"}),
        (async_then_sync, {"alphabeta", "betaalpha", "alphaalpha", "betabeta"}),
        (async_func_calling_sync_func, {"alphabeta", "betaalpha", "alphaalpha", "betabeta"}),
        (sync_func_calling_async_func, {"alphabeta", "betaalpha", "alphaalpha", "betabeta"}),
    ],
)
def test_async(gfunc, expected, sync):
    s = run(gfunc, sync)
    assert s in expected

@pytest.mark.parametrize(
    "sync",
    [
        True,
        False,
    ],
)
def test_async_with_sync_accessor(sync):
    # This should raise an AwaitException because the sync accessor is not
    # allowed in the async function
    with pytest.raises(AwaitException):
        run(async_func_with_sync_accessor, sync)

def test_sync_accessor_in_foreign_event_loop():
    async def main():
        lm = models.Mock()
        lm += sync_func()
        assert str(lm) in {"alpha", "beta"}
    asyncio.run(main())
