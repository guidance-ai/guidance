import pytest
from guidance._schema import GenTokenExtra, GenToken
from guidance.trace import TraceHandler, LiteralInput, TextOutput
from guidance.visual import TraceMessage, MetricMessage, ExecutionCompletedMessage, \
    TokensMessage, ResetDisplayMessage, ClientReadyMessage, OutputRequestMessage, \
    ClientReadyAckMessage, trace_node_to_html, display_trace_tree, trace_node_to_str
from guidance.visual import serialize_message, deserialize_message
from guidance.visual._async import async_loop, async_task, run_async_coroutine
from guidance.visual._environment import Environment
import asyncio


@pytest.mark.parametrize(
    "message",
    [
        TraceMessage(trace_id=0),
        MetricMessage(name="name", value="value"),
        ExecutionCompletedMessage(last_trace_id=0),
        TokensMessage(trace_id=0, text="text", tokens=[
            GenTokenExtra(token_id=0, prob=0, top_k=[GenToken(token_id=0, prob=0)])
        ]),
        ResetDisplayMessage(),
        ClientReadyMessage(),
        ClientReadyAckMessage(),
        OutputRequestMessage(),
    ]
)
def test_serialization(message):
    ser = serialize_message(message)
    deser = deserialize_message(ser)
    assert deser.model_dump() == message.model_dump()


def test_async():
    loop = async_loop()
    assert loop != asyncio.get_event_loop()

    async def f():
        return True

    task = run_async_coroutine(async_task(f())).result()
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
    assert display_trace_tree(trace_handler) is None


def test_environment():
    env = Environment()
    assert not env.is_cloud()
    assert not env.is_notebook()
    assert env.is_terminal()
    assert "ipython-zmq" not in env.detected_envs

