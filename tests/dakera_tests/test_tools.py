"""
Unit tests for DakeraTools.

Verifies that the three Tool objects are correctly constructed and that
their underlying callables hit the DakeraClient methods with the right args.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from guidance import Tool
from guidance_dakera._tools import DakeraTools


def _tools_with_mock(search_hits=None, store_result=None):
    """Return a DakeraTools instance whose _client is fully mocked."""
    dt = DakeraTools("test-agent")
    dt._client = MagicMock()
    dt._client.search.return_value = search_hits or []
    dt._client.store.return_value = store_result or {"memory": {"id": "t1"}}
    dt._client.forget.return_value = {}
    return dt


class TestDakeraToolsInit(unittest.TestCase):
    def test_default_top_k(self):
        dt = DakeraTools("a")
        self.assertEqual(dt.top_k, 5)

    def test_custom_params(self):
        dt = DakeraTools("a", top_k=20, fail_open=False)
        self.assertEqual(dt.top_k, 20)
        self.assertFalse(dt.fail_open)


class TestDakeraToolsProperties(unittest.TestCase):
    def setUp(self):
        self.dt = _tools_with_mock()

    def test_store_tool_is_guidance_tool(self):
        self.assertIsInstance(self.dt.store_tool, Tool)
        self.assertEqual(self.dt.store_tool.name, "dakera_store")

    def test_search_tool_is_guidance_tool(self):
        self.assertIsInstance(self.dt.search_tool, Tool)
        self.assertEqual(self.dt.search_tool.name, "dakera_search")

    def test_forget_tool_is_guidance_tool(self):
        self.assertIsInstance(self.dt.forget_tool, Tool)
        self.assertEqual(self.dt.forget_tool.name, "dakera_forget")

    def test_all_returns_three_tools(self):
        tools = self.dt.all()
        self.assertEqual(len(tools), 3)
        names = {t.name for t in tools}
        self.assertIn("dakera_store", names)
        self.assertIn("dakera_search", names)
        self.assertIn("dakera_forget", names)

    def test_read_only_returns_one_tool(self):
        tools = self.dt.read_only()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "dakera_search")


class TestDakeraToolsStore(unittest.TestCase):
    def test_store_callable(self):
        dt = _tools_with_mock(store_result={"memory": {"id": "new"}})
        result = dt._store("remember this", importance=0.8, tags="tag1,tag2")
        dt._client.store.assert_called_once_with(
            "remember this",
            "test-agent",
            session_id=None,
            importance=0.8,
            tags=["tag1", "tag2"],
        )
        self.assertIn("new", result)

    def test_store_empty_tags(self):
        dt = _tools_with_mock()
        dt._store("x", importance=0.5, tags="")
        call_kwargs = dt._client.store.call_args[1]
        self.assertIsNone(call_kwargs["tags"])

    def test_store_fail_open_returns_error_string(self):
        from guidance_dakera._client import DakeraConnectionError
        dt = DakeraTools("a", fail_open=True)
        dt._client = MagicMock()
        dt._client.store.side_effect = DakeraConnectionError("down")
        result = dt._store("x")
        self.assertIn("[Dakera error]", result)

    def test_store_fail_closed_raises(self):
        from guidance_dakera._client import DakeraConnectionError
        dt = DakeraTools("a", fail_open=False)
        dt._client = MagicMock()
        dt._client.store.side_effect = DakeraConnectionError("down")
        with self.assertRaises(DakeraConnectionError):
            dt._store("x")


class TestDakeraToolsSearch(unittest.TestCase):
    def test_search_formats_results(self):
        hits = [
            {"memory": {"id": "1", "content": "the sky is blue"}, "score": 0.95},
            {"memory": {"id": "2", "content": "water is wet"}, "score": 0.7},
        ]
        dt = _tools_with_mock(search_hits=hits)
        result = dt._search("weather")
        self.assertIn("the sky is blue", result)
        self.assertIn("water is wet", result)
        self.assertIn("0.95", result)

    def test_search_empty(self):
        dt = _tools_with_mock(search_hits=[])
        result = dt._search("nothing")
        self.assertEqual(result, "No relevant memories found.")

    def test_search_uses_default_top_k(self):
        dt = DakeraTools("a", top_k=7)
        dt._client = MagicMock()
        dt._client.search.return_value = []
        dt._search("q", top_k=0)  # 0 means use default
        dt._client.search.assert_called_once()
        call_kwargs = dt._client.search.call_args[1]
        self.assertEqual(call_kwargs["top_k"], 7)

    def test_search_overrides_top_k(self):
        dt = DakeraTools("a", top_k=5)
        dt._client = MagicMock()
        dt._client.search.return_value = []
        dt._search("q", top_k=3)
        call_kwargs = dt._client.search.call_args[1]
        self.assertEqual(call_kwargs["top_k"], 3)


class TestDakeraToolsForget(unittest.TestCase):
    def test_forget_all(self):
        dt = _tools_with_mock()
        result = dt._forget("")
        dt._client.forget.assert_called_once_with("test-agent", memory_ids=None)
        self.assertIn("All memories forgotten", result)

    def test_forget_specific_ids(self):
        dt = _tools_with_mock()
        result = dt._forget("id1, id2")
        dt._client.forget.assert_called_once_with("test-agent", memory_ids=["id1", "id2"])
        self.assertIn("id1", result)
        self.assertIn("id2", result)


class TestDakeraToolsCallableSchema(unittest.TestCase):
    """Verify that the Tool objects have properly inferred JSON schemas."""

    def setUp(self):
        self.dt = _tools_with_mock()

    def test_search_has_query_param(self):
        schema = self.dt.search_tool.tool.get_schema()
        self.assertIn("query", schema.get("properties", {}))

    def test_store_has_content_param(self):
        schema = self.dt.store_tool.tool.get_schema()
        self.assertIn("content", schema.get("properties", {}))

    def test_forget_has_memory_ids_param(self):
        schema = self.dt.forget_tool.tool.get_schema()
        self.assertIn("memory_ids", schema.get("properties", {}))


if __name__ == "__main__":
    unittest.main()
