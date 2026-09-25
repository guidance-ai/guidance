import types

from pydantic import BaseModel

from guidance import Tool


class Point(BaseModel):
    x: int
    y: int


def move_to(point: Point) -> str:
    """Move the robot to a point."""
    return f"moved to {point.x},{point.y}"


# Same function as above, but defined in a module that uses postponed (PEP 563)
# annotations, so `move_to`'s annotations are the raw strings "Point" and "str"
# at runtime and must be resolved in that module's namespace.
_POSTPONED_MODULE_SRC = """\
from __future__ import annotations

from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int


def move_to(point: Point) -> str:
    \"\"\"Move the robot to a point.\"\"\"
    return f"moved to {point.x},{point.y}"
"""


def test_from_callable():
    tool = Tool.from_callable(move_to)
    schema = tool.to_openai_style()["function"]["parameters"]
    assert schema["properties"]["point"] == {"$ref": "#/$defs/Point"}
    assert "Point" in schema["$defs"]


def test_from_callable_postponed_annotations():
    module = types.ModuleType("postponed_tools")
    exec(_POSTPONED_MODULE_SRC, module.__dict__)
    tool = Tool.from_callable(module.move_to)
    schema = tool.to_openai_style()["function"]["parameters"]
    assert schema["properties"]["point"] == {"$ref": "#/$defs/Point"}
    assert "Point" in schema["$defs"]
