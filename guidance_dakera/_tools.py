"""
Pre-built guidance.Tool objects for the Dakera memory API.

``DakeraTools`` produces three tool objects that can be passed directly to
``gen(..., tools=[...])`` — the standard guidance agentic-tool-call interface.

Usage::

    from guidance import gen, system, user, assistant, models
    from guidance_dakera import DakeraTools

    lm = models.OpenAI("gpt-4o-mini")
    tools = DakeraTools(agent_id="research-bot")

    with system():
        lm += "You are a research assistant with persistent memory."

    with user():
        lm += "What did you learn about neural scaling laws?"

    with assistant():
        lm += gen(tools=tools.all(), max_tokens=500)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from guidance import Tool

from ._client import DakeraClient

logger = logging.getLogger(__name__)


class DakeraTools:
    """Factory for guidance ``Tool`` objects backed by a Dakera server.

    Parameters
    ----------
    agent_id:
        Stable identifier for the agent (all tool calls use this scope).
    base_url:
        Dakera server URL.
    api_key:
        Bearer token.
    session_id:
        Optional session scope for every call.
    top_k:
        Default top_k for the search tool (can be overridden per call).
    fail_open:
        If ``True`` (default), errors return an error string rather than
        raising, so the model can handle failures gracefully.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
        fail_open: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.top_k = top_k
        self.fail_open = fail_open
        self._client = DakeraClient(base_url=base_url, api_key=api_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _safe(self, fn, *args, **kwargs) -> str:
        """Execute fn(*args, **kwargs), returning an error string on failure."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = f"[Dakera error] {exc}"
            if self.fail_open:
                logger.warning(msg)
                return msg
            raise

    # ------------------------------------------------------------------
    # Tool callables
    # ------------------------------------------------------------------

    def _store(self, content: str, importance: float = 0.5, tags: str = "") -> str:
        """Store *content* in Dakera memory.

        Args:
            content: The text to remember.
            importance: A value 0.0–1.0 indicating how important this memory
                is (higher = decays more slowly).
            tags: Comma-separated tags to attach, e.g. ``"research,papers"``.
        """
        def _do():
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
            result = self._client.store(
                content,
                self.agent_id,
                session_id=self.session_id,
                importance=importance,
                tags=tag_list,
            )
            mem_id = result.get("memory", {}).get("id", "?")
            return f"Memory stored successfully (id={mem_id})."
        return self._safe(_do)

    def _search(self, query: str, top_k: int = 0) -> str:
        """Search Dakera for memories relevant to *query*.

        Args:
            query: Natural-language search query.
            top_k: Number of results to return (0 = use server default).
        """
        def _do():
            k = top_k if top_k > 0 else self.top_k
            hits = self._client.search(
                query,
                self.agent_id,
                top_k=k,
                session_id=self.session_id,
            )
            if not hits:
                return "No relevant memories found."
            parts = []
            for i, hit in enumerate(hits, 1):
                mem = hit.get("memory", {})
                score = hit.get("score", 0.0)
                content = mem.get("content", "").strip()
                mem_id = mem.get("id", "?")
                parts.append(f"{i}. [score={score:.2f}, id={mem_id}] {content}")
            return "\n".join(parts)
        return self._safe(_do)

    def _forget(self, memory_ids: str = "") -> str:
        """Forget memories from Dakera.

        Args:
            memory_ids: Comma-separated memory IDs to delete.  Leave empty
                to delete **all** memories for this agent.
        """
        def _do():
            ids = [m.strip() for m in memory_ids.split(",") if m.strip()] if memory_ids else None
            self._client.forget(self.agent_id, memory_ids=ids)
            if ids:
                return f"Forgotten {len(ids)} memories: {', '.join(ids)}."
            return "All memories forgotten for this agent."
        return self._safe(_do)

    # ------------------------------------------------------------------
    # Tool objects
    # ------------------------------------------------------------------

    @property
    def store_tool(self) -> Tool:
        """A :class:`guidance.Tool` that stores a memory."""
        return Tool.from_callable(
            self._store,
            name="dakera_store",
            description=(
                "Store information in persistent memory for later retrieval. "
                "Use this when you learn something important that should be "
                "remembered across conversations."
            ),
        )

    @property
    def search_tool(self) -> Tool:
        """A :class:`guidance.Tool` that searches memories."""
        return Tool.from_callable(
            self._search,
            name="dakera_search",
            description=(
                "Search persistent memory for information relevant to a query. "
                "Use this when the user asks about something you might have "
                "previously learned or been told."
            ),
        )

    @property
    def forget_tool(self) -> Tool:
        """A :class:`guidance.Tool` that forgets specific memories."""
        return Tool.from_callable(
            self._forget,
            name="dakera_forget",
            description=(
                "Delete specific memories (by ID) or all memories for this "
                "agent.  Use when the user explicitly asks you to forget "
                "something, or when stored information is known to be wrong."
            ),
        )

    def all(self) -> list[Tool]:
        """Return all three tools as a list ready for ``gen(tools=...)``.

        Example::

            with assistant():
                lm += gen(tools=tools.all(), max_tokens=500)
        """
        return [self.search_tool, self.store_tool, self.forget_tool]

    def read_only(self) -> list[Tool]:
        """Return only the search tool (safe for read-only contexts)."""
        return [self.search_tool]
