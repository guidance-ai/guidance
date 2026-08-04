"""Regression tests for guidance._tools.Tool.call error handling.

See https://github.com/guidance-ai/guidance/issues/1485: ``Tool.call`` used to
catch ``BaseException`` (swallowing ``KeyboardInterrupt``/``SystemExit`` and
turning them into tool-output strings) and asserted that the traceback always
had an inner frame. That assert fired whenever the error was raised at the call
expression itself rather than inside the callable -- most reachably when the
provider returns an argument set that does not match the tool's signature, which
raises ``TypeError`` before the callable's frame is ever entered.
"""

import pytest
from guidance._tools import Tool


def _tool(fn):
    return Tool.from_callable(fn, name="t", description="d")


def test_call_propagates_keyboard_interrupt():
    def raises_keyboard_interrupt():
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _tool(raises_keyboard_interrupt).call()


def test_call_propagates_system_exit():
    def raises_system_exit():
        raise SystemExit(2)

    with pytest.raises(SystemExit):
        _tool(raises_system_exit).call()


def test_call_formats_regular_exception():
    def raises_value_error():
        raise ValueError("kaboom")

    result = _tool(raises_value_error).call()
    assert isinstance(result, str)
    assert "ValueError: kaboom" in result
    # The traceback should start inside the callable, not inside Tool.call.
    assert "raises_value_error" in result


def test_call_with_mismatched_arguments_is_formatted_not_asserted():
    def get_weather(city: str, units: str = "c"):
        return f"{city}:{units}"

    tool = _tool(get_weather)

    # A provider can return an argument dict that does not match the signature; it is
    # splatted straight into Tool.call, so the TypeError is raised at the call
    # expression and never enters get_weather's frame.
    missing_required = tool.call()
    assert isinstance(missing_required, str)
    assert "missing 1 required positional argument" in missing_required

    unexpected = tool.call(city="Rome", zoom=3)
    assert isinstance(unexpected, str)
    assert "unexpected keyword argument" in unexpected

    # A well-formed call is unaffected.
    assert tool.call(city="Rome") == "Rome:c"


def test_call_non_callable_reports_type_error():
    class NotCallable:
        pass

    tool = _tool(lambda: None)
    # Simulate a non-callable value assigned to .callable: the TypeError is then
    # raised at the call expression itself, so there is no inner traceback frame.
    tool.callable = NotCallable()
    result = tool.call()
    assert isinstance(result, str)
    assert "not callable" in result
