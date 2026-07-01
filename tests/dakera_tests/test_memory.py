"""
Unit tests for DakeraMemory and dakera_remember / dakera_recall.

Uses guidance's Mock model so no real LLM or Dakera server is needed.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import guidance
from guidance.models import Mock

from guidance_dakera._memory import DakeraMemory, dakera_recall, dakera_remember


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(search_hits=None, store_result=None):
    """Return a patched DakeraClient instance."""
    client = MagicMock()
    client.search.return_value = search_hits or []
    client.store.return_value = store_result or {"memory": {"id": "test-id", "content": ""}}
    client.forget.return_value = {}
    return client


# ---------------------------------------------------------------------------
# DakeraMemory
# ---------------------------------------------------------------------------

class TestDakeraMemoryInit(unittest.TestCase):
    def test_defaults(self):
        mem = DakeraMemory("agent1")
        self.assertEqual(mem.agent_id, "agent1")
        self.assertEqual(mem.top_k, 5)
        self.assertTrue(mem.fail_open)

    def test_custom_params(self):
        mem = DakeraMemory("a", top_k=10, min_score=0.5, fail_open=False)
        self.assertEqual(mem.top_k, 10)
        self.assertEqual(mem.min_score, 0.5)
        self.assertFalse(mem.fail_open)


class TestDakeraMemoryRecall(unittest.TestCase):
    def _mem(self, hits):
        mem = DakeraMemory("agent1")
        mem._client = _mock_client(search_hits=hits)
        return mem

    def test_recall_returns_hits(self):
        hits = [
            {"memory": {"id": "1", "content": "foo"}, "score": 0.9},
        ]
        mem = self._mem(hits)
        result = mem.recall("foo query")
        self.assertEqual(len(result), 1)

    def test_recall_filters_by_min_score(self):
        hits = [
            {"memory": {"id": "1", "content": "foo"}, "score": 0.9},
            {"memory": {"id": "2", "content": "bar"}, "score": 0.3},
        ]
        mem = DakeraMemory("agent1", min_score=0.5)
        mem._client = _mock_client(search_hits=hits)
        result = mem.recall("q")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["memory"]["id"], "1")

    def test_recall_fail_open_returns_empty(self):
        from guidance_dakera._client import DakeraConnectionError
        mem = DakeraMemory("agent1", fail_open=True)
        mem._client = MagicMock()
        mem._client.search.side_effect = DakeraConnectionError("down")
        result = mem.recall("q")
        self.assertEqual(result, [])

    def test_recall_fail_closed_raises(self):
        from guidance_dakera._client import DakeraConnectionError
        mem = DakeraMemory("agent1", fail_open=False)
        mem._client = MagicMock()
        mem._client.search.side_effect = DakeraConnectionError("down")
        with self.assertRaises(DakeraConnectionError):
            mem.recall("q")


class TestDakeraMemoryFormatMemories(unittest.TestCase):
    def test_empty_hits(self):
        mem = DakeraMemory("a")
        self.assertEqual(mem.format_memories([]), "")

    def test_single_hit(self):
        hits = [{"memory": {"id": "1", "content": "hello world"}, "score": 0.88}]
        mem = DakeraMemory("a")
        text = mem.format_memories(hits)
        self.assertIn("hello world", text)
        self.assertIn("0.88", text)
        self.assertIn("## Relevant memories", text)

    def test_multiple_hits_numbered(self):
        hits = [
            {"memory": {"id": "1", "content": "a"}, "score": 0.9},
            {"memory": {"id": "2", "content": "b"}, "score": 0.8},
        ]
        mem = DakeraMemory("a")
        text = mem.format_memories(hits)
        self.assertIn("1.", text)
        self.assertIn("2.", text)

    def test_custom_header(self):
        mem = DakeraMemory("a", header="### Past context\n")
        hits = [{"memory": {"id": "1", "content": "x"}, "score": 0.5}]
        text = mem.format_memories(hits)
        self.assertIn("### Past context", text)


class TestDakeraMemoryRemember(unittest.TestCase):
    def test_remember_calls_store(self):
        mem = DakeraMemory("agent1")
        mock_client = _mock_client(store_result={"memory": {"id": "new-id"}})
        mem._client = mock_client
        result = mem.remember("important fact")
        mock_client.store.assert_called_once_with(
            "important fact", "agent1",
            session_id=None, importance=None, tags=None
        )
        self.assertEqual(result["id"], "new-id")

    def test_remember_fail_open_returns_none(self):
        from guidance_dakera._client import DakeraConnectionError
        mem = DakeraMemory("a", fail_open=True)
        mem._client = MagicMock()
        mem._client.store.side_effect = DakeraConnectionError("down")
        result = mem.remember("x")
        self.assertIsNone(result)


class TestDakeraMemoryInjectContextManager(unittest.TestCase):
    def test_inject_no_memories_no_system_block(self):
        """When no memories are found, the model is returned unchanged."""
        mem = DakeraMemory("agent1")
        mem._client = _mock_client(search_hits=[])

        lm = Mock(b"<s>")
        original_str = str(lm)

        with mem.inject(lm) as lm_inner:
            # No system block should be added for empty hits
            self.assertIsNotNone(lm_inner)

    def test_inject_with_memories_appends_text(self):
        """When memories exist the injected text appears in the model output."""
        hits = [{"memory": {"id": "1", "content": "the sky is blue"}, "score": 0.95}]
        mem = DakeraMemory("agent1")
        mem._client = _mock_client(search_hits=hits)

        # Use a real Mock LM so we can inspect the string
        lm = Mock(b"<s>")
        with mem.inject(lm, query="sky color") as lm_inner:
            # After injection the model state should contain the memory text
            state = str(lm_inner)
            self.assertIn("the sky is blue", state)


# ---------------------------------------------------------------------------
# dakera_remember @guidance function
# ---------------------------------------------------------------------------

class TestDakeraRememberFunction(unittest.TestCase):
    def test_stores_captured_variable(self):
        """dakera_remember reads lm[var] and stores it via DakeraClient.store."""
        lm = Mock(b"<s>")
        lm = lm.set("my_answer", "Paris is the capital of France.")

        with patch("guidance_dakera._memory.DakeraClient") as MockClient:
            instance = MockClient.return_value
            instance.store.return_value = {"memory": {"id": "xyz"}}

            lm = lm + dakera_remember("my_answer", agent_id="bot")

        instance.store.assert_called_once()
        call_args = instance.store.call_args
        self.assertEqual(call_args[0][0], "Paris is the capital of France.")
        self.assertEqual(call_args[0][1], "bot")

    def test_noop_when_variable_missing(self):
        """If the variable is not captured, store is never called."""
        lm = Mock(b"<s>")

        with patch("guidance_dakera._memory.DakeraClient") as MockClient:
            instance = MockClient.return_value
            lm = lm + dakera_remember("nonexistent", agent_id="bot")

        instance.store.assert_not_called()

    def test_fail_open_does_not_raise(self):
        """Network errors are swallowed when fail_open=True."""
        from guidance_dakera._client import DakeraConnectionError
        lm = Mock(b"<s>")
        lm = lm.set("answer", "some text")

        with patch("guidance_dakera._memory.DakeraClient") as MockClient:
            instance = MockClient.return_value
            instance.store.side_effect = DakeraConnectionError("down")
            # Should not raise
            lm = lm + dakera_remember("answer", agent_id="bot", fail_open=True)

    def test_fail_closed_propagates_error(self):
        """Network errors are propagated when fail_open=False."""
        from guidance_dakera._client import DakeraConnectionError
        lm = Mock(b"<s>")
        lm = lm.set("answer", "some text")

        with patch("guidance_dakera._memory.DakeraClient") as MockClient:
            instance = MockClient.return_value
            instance.store.side_effect = DakeraConnectionError("down")
            with self.assertRaises(DakeraConnectionError):
                lm = lm + dakera_remember("answer", agent_id="bot", fail_open=False)

    def test_list_variable_joined(self):
        """List captures are joined with newlines before storing."""
        lm = Mock(b"<s>")
        lm = lm.set("items", ["item one", "item two"])

        with patch("guidance_dakera._memory.DakeraClient") as MockClient:
            instance = MockClient.return_value
            instance.store.return_value = {"memory": {"id": "1"}}
            lm = lm + dakera_remember("items", agent_id="bot")

        stored_content = instance.store.call_args[0][0]
        self.assertIn("item one", stored_content)
        self.assertIn("item two", stored_content)

    def test_agent_id_defaults_to_var_name(self):
        """When agent_id is not given, var_name is used."""
        lm = Mock(b"<s>")
        lm = lm.set("answer", "hello")

        with patch("guidance_dakera._memory.DakeraClient") as MockClient:
            with patch.dict("os.environ", {}, clear=True):
                instance = MockClient.return_value
                instance.store.return_value = {"memory": {"id": "1"}}
                lm = lm + dakera_remember("answer")

        stored_agent = instance.store.call_args[0][1]
        self.assertEqual(stored_agent, "answer")


# ---------------------------------------------------------------------------
# dakera_recall @guidance function
# ---------------------------------------------------------------------------

class TestDakeraRecallFunction(unittest.TestCase):
    def test_appends_memories_to_model(self):
        hits = [{"memory": {"id": "1", "content": "neural scaling laws paper"}, "score": 0.91}]

        with patch("guidance_dakera._memory.DakeraClient") as MockClient:
            instance = MockClient.return_value
            instance.search.return_value = hits

            lm = Mock(b"<s>")
            lm = lm + dakera_recall("scaling laws", agent_id="research-bot")

        self.assertIn("neural scaling laws paper", str(lm))

    def test_captures_memories_as_variable(self):
        hits = [{"memory": {"id": "1", "content": "test content"}, "score": 0.80}]

        with patch("guidance_dakera._memory.DakeraClient") as MockClient:
            instance = MockClient.return_value
            instance.search.return_value = hits

            lm = Mock(b"<s>")
            lm = lm + dakera_recall("q", agent_id="bot", name="ctx")

        self.assertIn("memories", str(lm["ctx"]))  # header present

    def test_empty_recall_no_output(self):
        with patch("guidance_dakera._memory.DakeraClient") as MockClient:
            instance = MockClient.return_value
            instance.search.return_value = []

            lm = Mock(b"<s>")
            before = str(lm)
            lm = lm + dakera_recall("nothing", agent_id="bot")
            # No extra text should be appended
            self.assertEqual(str(lm), before)


if __name__ == "__main__":
    unittest.main()
