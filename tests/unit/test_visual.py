import pytest
from guidance._schema import GenTokenExtra, GenToken
from guidance.trace import TraceHandler, LiteralInput, TextOutput
from guidance.visual import TraceMessage, MetricMessage, ExecutionCompletedMessage, \
    ExecutionCompletedOutputMessage, ResetDisplayMessage, ClientReadyMessage, OutputRequestMessage, \
    ClientReadyAckMessage, trace_node_to_html
from guidance.visual import serialize_message, deserialize_message
from guidance.visual._async import async_loop, async_task, run_async_coroutine
import asyncio


@pytest.mark.parametrize(
    "message",
    [
        TraceMessage(trace_id=0),
        MetricMessage(name="name", value="value"),
        ExecutionCompletedMessage(last_trace_id=0),
        ExecutionCompletedOutputMessage(trace_id=0, text="text", tokens=[
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


def test_str_methods():
    trace_handler = TraceHandler()
    root_node = trace_handler.update_node(0, None, None)
    trace_handler.update_node(1, 0, None)
    inp = LiteralInput(value="")
    out = TextOutput(value="")
    trace_handler.update_node(2, 0, inp)
    child_node = trace_handler.update_node(2, 0, out)

    assert trace_node_to_html(child_node) != ""