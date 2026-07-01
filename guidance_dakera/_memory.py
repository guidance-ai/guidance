"""
High-level guidance integration for Dakera memory.

Two integration styles are provided:

1. **DakeraMemory** — a Python context manager that wraps a guidance
   program turn.  Before the body executes it retrieves relevant
   memories from Dakera and prepends them as a system block; after
   the body executes you can call ``remember()`` to persist new content.

2. **dakera_remember()** — a ``@guidance``-decorated function you
   can append to any model (``lm += dakera_remember("var_name")``).
   It reads a captured variable from the model, stores it in Dakera,
   and returns the model unchanged so execution continues normally.
"""

from __future__ import annotations

import logging
import textwrap
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, Optional

from guidance import guidance as guidance_decorator
from guidance import system

from ._client import DakeraClient, DakeraConnectionError

if TYPE_CHECKING:
    from guidance.models._base._model import Model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DakeraMemory
# ---------------------------------------------------------------------------

class DakeraMemory:
    """Retrieve and persist memories around a guidance program turn.

    Parameters
    ----------
    agent_id:
        Stable identifier for the agent (scopes all memory operations).
    base_url:
        Dakera server URL (default: ``DAKERA_BASE_URL`` env var or
        ``http://localhost:3300``).
    api_key:
        Bearer token (default: ``DAKERA_API_KEY`` env var).
    session_id:
        Optional session scope passed on every request.
    top_k:
        How many memories to surface in the context (default 5).
    min_score:
        Minimum cosine-similarity score to include a memory (default 0.0,
        i.e. all results from the server).
    header:
        Text prepended before the memory block in the system message.
    fail_open:
        If ``True`` (default), network errors are logged as warnings but
        execution continues.  If ``False``, errors propagate.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.0,
        header: str = "## Relevant memories\n",
        fail_open: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.top_k = top_k
        self.min_score = min_score
        self.header = header
        self.fail_open = fail_open
        self._client = DakeraClient(base_url=base_url, api_key=api_key)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def recall(self, query: str) -> list[dict]:
        """Return raw memory records for ``query`` (best-effort)."""
        try:
            hits = self._client.search(
                query,
                self.agent_id,
                top_k=self.top_k,
                session_id=self.session_id,
            )
            return [h for h in hits if h.get("score", 1.0) >= self.min_score]
        except Exception as exc:
            if self.fail_open:
                logger.warning("Dakera recall failed (fail_open=True): %s", exc)
                return []
            raise

    def format_memories(self, hits: list[dict]) -> str:
        """Convert a list of ``{memory, score}`` hits to an injection string."""
        if not hits:
            return ""
        lines = [self.header]
        for i, hit in enumerate(hits, 1):
            mem = hit.get("memory", {})
            content = mem.get("content", "").strip()
            score = hit.get("score", 0.0)
            lines.append(f"{i}. [{score:.2f}] {content}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def remember(
        self,
        content: str,
        *,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Persist *content* to Dakera (best-effort).

        Returns the created memory record or ``None`` on error (fail_open).
        """
        try:
            result = self._client.store(
                content,
                self.agent_id,
                session_id=self.session_id,
                importance=importance,
                tags=tags,
            )
            mem = result.get("memory", {})
            logger.debug("Stored memory id=%s", mem.get("id"))
            return mem
        except Exception as exc:
            if self.fail_open:
                logger.warning("Dakera store failed (fail_open=True): %s", exc)
                return None
            raise

    def forget(self, memory_ids: Optional[list[str]] = None) -> None:
        """Delete memories from Dakera (best-effort)."""
        try:
            self._client.forget(self.agent_id, memory_ids=memory_ids)
        except Exception as exc:
            if self.fail_open:
                logger.warning("Dakera forget failed (fail_open=True): %s", exc)
            else:
                raise

    # ------------------------------------------------------------------
    # Context manager — inject memories before a guidance turn
    # ------------------------------------------------------------------

    @contextmanager
    def inject(
        self,
        lm: "Model",
        query: str = "",
        *,
        importance: Optional[float] = None,
        tags: Optional[list[str]] = None,
    ) -> Generator["Model", None, None]:
        """Context manager that prepends relevant memories into the system block.

        Usage::

            mem = DakeraMemory(agent_id="bot")

            with system():
                lm += "You are a helpful assistant."

            with mem.inject(lm, query="what did we talk about") as lm:
                with user():
                    lm += "Remind me what we discussed."
                with assistant():
                    lm += gen(name="reply", max_tokens=200)

            # After the with-block, persist the assistant turn:
            mem.remember(lm["reply"])

        The ``query`` parameter guides which memories to fetch.  If omitted
        the last ``str(lm)`` tail (up to 512 chars) is used as the query.
        """
        effective_query = query or str(lm)[-512:]
        hits = self.recall(effective_query)
        block_text = self.format_memories(hits)

        if block_text:
            with system():
                lm += block_text

        yield lm


# ---------------------------------------------------------------------------
# dakera_remember — @guidance function
# ---------------------------------------------------------------------------

@guidance_decorator(stateless=False)
def dakera_remember(
    lm: "Model",
    var_name: str,
    *,
    agent_id: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
    importance: Optional[float] = None,
    tags: Optional[list[str]] = None,
    fail_open: bool = True,
) -> "Model":
    """Guidance function that stores a captured variable to Dakera.

    Append this after any generation that captures a variable you want to
    remember::

        lm += gen(name="answer", max_tokens=200)
        lm += dakera_remember("answer", agent_id="my-agent")

    Parameters
    ----------
    var_name:
        The name of a captured guidance variable (``lm[var_name]``).
        If the variable does not exist the function is a no-op.
    agent_id:
        Overrides ``DAKERA_AGENT_ID`` env var (both fall back to the
        ``var_name`` string so the function is always safe to call).
    base_url / api_key:
        Passed directly to :class:`DakeraClient`.
    session_id:
        Optional session scope.
    importance:
        Importance weight (0.0–1.0) passed to the store endpoint.
    tags:
        List of tags attached to the memory record.
    fail_open:
        If ``True`` (default), errors are logged and execution continues.
    """
    import os as _os

    effective_agent_id = (
        agent_id
        or _os.environ.get("DAKERA_AGENT_ID")
        or var_name
    )

    content = lm.get(var_name)
    if not content:
        logger.debug("dakera_remember: variable %r is empty/missing, skipping.", var_name)
        return lm

    if isinstance(content, list):
        content = "\n".join(content)

    client = DakeraClient(base_url=base_url, api_key=api_key)
    try:
        result = client.store(
            content,
            effective_agent_id,
            session_id=session_id,
            importance=importance,
            tags=tags,
        )
        mem = result.get("memory", {})
        logger.debug("dakera_remember: stored memory id=%s", mem.get("id"))
    except Exception as exc:
        if fail_open:
            logger.warning("dakera_remember failed (fail_open=True): %s", exc)
        else:
            raise

    return lm


# ---------------------------------------------------------------------------
# dakera_recall — @guidance function (injects memories as a string)
# ---------------------------------------------------------------------------

@guidance_decorator(stateless=False)
def dakera_recall(
    lm: "Model",
    query: str,
    *,
    name: str = "memories",
    agent_id: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.0,
    header: str = "## Relevant memories\n",
    fail_open: bool = True,
) -> "Model":
    """Guidance function that retrieves memories and appends them to the model.

    The resulting text is also captured in ``lm[name]`` so downstream
    guidance functions can reference it::

        lm += dakera_recall("last conversation", agent_id="bot", name="ctx")
        lm += "The assistant recalled: " + lm["ctx"]
    """
    import os as _os

    effective_agent_id = (
        agent_id
        or _os.environ.get("DAKERA_AGENT_ID")
        or "default"
    )

    mem_obj = DakeraMemory(
        effective_agent_id,
        base_url=base_url,
        api_key=api_key,
        session_id=session_id,
        top_k=top_k,
        min_score=min_score,
        header=header,
        fail_open=fail_open,
    )
    hits = mem_obj.recall(query)
    block_text = mem_obj.format_memories(hits)

    lm += block_text
    lm = lm.set(name, block_text)
    return lm
