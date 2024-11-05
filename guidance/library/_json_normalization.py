from __future__ import annotations
from typing import Any, Optional
from itertools import product


def normalize_allOf(subnodes: list[dict[str, Any]], siblings: dict[str, Any] = {}) -> dict[str, Any]:
    if not subnodes:
        return siblings

    # Normalization will ensure that there are no applicable "anyOf" or "oneOf" keys
    # except at the top level of the schema
    subnodes = [normalize(node) for node in subnodes]
    groups = []
    if any("oneOf" in node for node in subnodes):
        # Binds more tightly than anyOf
        kind = "oneOf"
    elif any("anyOf" in node for node in subnodes):
        kind = "anyOf"
    else:
        # We are done
        return maybe_allOf(subnodes, siblings)

    other = []
    if siblings:
        other.append(siblings)

    for node in subnodes:
        if "oneOf" in node and "anyOf" in node:
            oneOf_list = node.pop("oneOf")
            anyOf_list = node.pop("anyOf")
            groups.append(list(product(oneOf_list, anyOf_list)))

        elif "oneOf" in node:
            oneOf_list = node.pop("oneOf")
            groups.append(oneOf_list)

        elif "anyOf" in node:
            anyOf_list = node.pop("anyOf")
            groups.append(anyOf_list)

        if "allOf" in node:
            other.extend(node.pop("allOf"))

        if node:
            # If there are any keys left, they need to end up in every allOf
            other.append(node)

    return {
        kind: [
            maybe_allOf([*item, *other])
            for item in product(*groups)
        ]
    }

def maybe_allOf(nodes: list[dict[str, Any]], siblings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if len(nodes) == 1 and not siblings:
        return nodes[0]
    if siblings:
        return {"allOf": [*nodes, siblings]}
    return {"allOf": nodes}


def normalize(node: dict[str, Any]) -> dict[str, Any]:
    node = normalize_allOf(node.pop("allOf", []), node)
    oneOf_list = node.pop("oneOf", [])
    anyOf_list = node.pop("anyOf", [])
    allOf_list = node.pop("allOf", [])

    if oneOf_list and anyOf_list:
        node = {
            "oneOf": [
                maybe_allOf([oneOf_item, anyOf_item, *allOf_list], node)
                for anyOf_item in anyOf_list
                for oneOf_item in oneOf_list
            ]
        }
    elif oneOf_list:
        node = {
            "oneOf": [
                maybe_allOf([oneOf_item, *allOf_list], node)
                for oneOf_item in oneOf_list
            ]
        }
    elif anyOf_list:
        node = {
            "anyOf": [
                maybe_allOf([anyOf_item, *allOf_list], node)
                for anyOf_item in anyOf_list
            ]
        }
    elif allOf_list:
        node = maybe_allOf([node, *allOf_list])
    return node

def normalize_oneOf_anyOf(node: dict[str, Any]) -> dict[str, Any]:
    oneOf_list = node.pop("oneOf", [])
    anyOf_list = node.pop("anyOf", [])
    allOf_list = node.pop("allOf", [])

    if oneOf_list and anyOf_list:
        node = {
            "oneOf": [
                maybe_allOf([oneOf_item, anyOf_item, *allOf_list], node)
                for anyOf_item in anyOf_list
                for oneOf_item in oneOf_list
            ]
        }
    elif oneOf_list:
        node = {
            "oneOf": [
                maybe_allOf([oneOf_item, *allOf_list], node)
                for oneOf_item in oneOf_list
            ]
        }
    elif anyOf_list:
        node = {
            "anyOf": [
                maybe_allOf([anyOf_item, *allOf_list], node)
                for anyOf_item in anyOf_list
            ]
        }
