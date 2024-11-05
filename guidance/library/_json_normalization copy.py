from __future__ import annotations
from typing import Any, Optional, TypedDict, cast, NamedTuple
from itertools import product

from typing import TypedDict, List, Union, Any, Dict

# Unnormalized Schema Definitions

class BaseSchema(TypedDict, total=False):
    type: str
    properties: Dict[str, Any]
    items: Any
    required: List[str]
    enum: List[Any]
    const: Any
    minimum: int
    maximum: int
    minLength: int
    maxLength: int
    pattern: str
    # Other schema keywords can be added here

class Schema(BaseSchema):
    allOf: List[Schema]
    anyOf: List[Schema]
    oneOf: List[Schema]

# Normalized Schema Definitions

class NormalizedAllOfSchema(BaseSchema):
    allOf: List[BaseSchema]

class NormalizedAnyOfSchema(TypedDict):
    anyOf: List[Union[NormalizedAllOfSchema, BaseSchema]]

class NormalizedOneOfSchema(TypedDict):
    oneOf: List[Union[NormalizedAllOfSchema, BaseSchema]]

# The NormalizedSchema can be a NormalizedBaseSchema or top-level combinators without nesting
NormalizedSchema = Union[BaseSchema, NormalizedAllOfSchema, NormalizedAnyOfSchema, NormalizedOneOfSchema]

class Combinators(NamedTuple):
    allOf: List[Schema]
    anyOf: List[Schema]
    oneOf: List[Schema]

def maybe_allOf(nodes: list[BaseSchema],  siblings: Optional[BaseSchema] = None) -> NormalizedSchema:
    if len(nodes) == 1 and not siblings:
        return nodes[0]
    if siblings:
        return {"allOf": [*nodes, siblings]}
    return {"allOf": nodes}

def get_combinators_and_siblings(node: Schema) -> tuple[Combinators, BaseSchema]:
    allOf = cast(list[Schema], node.pop("allOf", []))
    oneOf = cast(list[Schema], node.pop("oneOf", []))
    anyOf = cast(list[Schema], node.pop("anyOf", []))
    siblings = node
    return Combinators(allOf, oneOf, anyOf), siblings

def normalize(orig_node: Schema) -> NormalizedSchema:
    ((allOf_list, oneOf_list, anyOf_list), siblings) = get_combinators_and_siblings(orig_node)
    if not allOf_list and not oneOf_list and not anyOf_list:
        return siblings
    
    allOf_list = normalize_allOf(allOf_list, siblings)
    anyOf_list = normalize_oneOf_anyOf(anyOf_list)

    if oneOf_list and anyOf_list:
        node: NormalizedOneOfSchema = {
            "oneOf": [
                maybe_allOf([oneOf_item, anyOf_item, *allOf_list])
                for anyOf_item in anyOf_list
                for oneOf_item in oneOf_list
            ]
        }
    elif oneOf_list:
        node: NormalizedOneOfSchema = {
            "oneOf": [
                maybe_allOf([oneOf_item, *allOf_list])
                for oneOf_item in oneOf_list
            ]
        }
    elif anyOf_list:
        node: NormalizedAnyOfSchema = {
            "anyOf": [
                maybe_allOf([anyOf_item, *allOf_list])
                for anyOf_item in anyOf_list
            ]
        }
    elif allOf_list:
        node: NormalizedSchema = maybe_allOf(allOf_list)
    
    return node