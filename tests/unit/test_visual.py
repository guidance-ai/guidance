import asyncio
from base64 import b64encode
import sys

import pytest

from guidance.registry import get_bg_async
from guidance.trace import Backtrack, LiteralInput, TextOutput, Token, TokenOutput, TraceHandler
from guidance.visual import (
    ClientReadyAckMessage,
    ClientReadyMessage,
    ExecutionCompletedMessage,
    GuidanceMessage,
    MetricMessage,
    OutputRequestMessage,
    ResetDisplayMessage,
    TopicExchange,
    TraceMessage,
    deserialize_message,
    display_trace_tree,
    serialize_message,
    trace_node_to_html,
    trace_node_to_str,
)
from guidance.visual._environment import Environment


@pytest.mark.parametrize(
    "message",
    [
        TraceMessage(trace_id=0),
        TraceMessage(
            trace_id=1, node_attr=TokenOutput(value="text", token=Token(token="text", bytes=b64encode(b"text"), prob=0))
        ),
        TraceMessage(trace_id=2, node_attr=Backtrack(n_tokens=1, bytes=b"")),
        MetricMessage(name="name", value="value"),
        ExecutionCompletedMessage(last_trace_id=0),
        ResetDisplayMessage(),
        ClientReadyMessage(),
        ClientReadyAckMessage(),
        OutputRequestMessage(),
    ],
)
def test_serialization(message):
    ser = serialize_message(message)
    deser = deserialize_message(ser)
    assert deser.model_dump() == message.model_dump()


def test_async():
    _, loop = get_bg_async()._thread_and_loop()
    if sys.version_info < (3, 14):
        assert loop != asyncio.get_event_loop()
    else:
        # python 3.14 made asyncio.get_event_loop() a RuntimeError
        # if there is no current event loop
        with pytest.raises(RuntimeError):
            asyncio.get_event_loop()

    async def f():
        return True

    task = get_bg_async().run_async_coroutine(get_bg_async().async_task(f())).result()
    assert task.result() is True


def test_str_method_smoke():
    trace_handler = TraceHandler()
    trace_handler.update_node(1, 0, None)
    inp = LiteralInput(value="Hi there!")
    out = TextOutput(value="Hi there!")
    trace_handler.update_node(2, 0, inp)
    child_node = trace_handler.update_node(2, 0, out)

    assert trace_node_to_html(child_node) != ""
    assert trace_node_to_str(child_node) != ""

    try:
        assert display_trace_tree(trace_handler) is None
    except ImportError:  # NOTE(nopdive): anytree not installed
        pass
    except Exception as e:
        raise e


def test_environment():
    env = Environment()
    assert not env.is_cloud()
    assert not env.is_notebook()
    assert env.is_terminal()
    assert "ipython-zmq" not in env.detected_envs


def test_exchange():
    exchange = TopicExchange()
    assert len(exchange._observers) == 0

    count = 0

    def inc(_: GuidanceMessage):
        nonlocal count
        count += 1

    # Defaults
    exchange.subscribe(inc)
    exchange.publish(GuidanceMessage())
    exchange.unsubscribe(inc)
    assert count == 1
    assert len(exchange._observers) == 0

    # Topic pattern set
    topic_pat = "no"
    exchange.subscribe(inc, topic_pat)
    exchange.publish(GuidanceMessage(), topic_pat)
    exchange.unsubscribe(inc, topic_pat)
    assert count == 2
    assert len(exchange._observers) == 0

    # Missed topic
    topic_pat = "what"
    exchange.subscribe(inc, topic_pat)
    exchange.publish(GuidanceMessage())
    exchange.unsubscribe(inc, topic_pat)
    assert count == 2
    assert len(exchange._observers) == 0
