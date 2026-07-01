"""
guidance-dakera: Dakera persistent vector memory for Guidance programs.

Provides three integration points:

1. DakeraMemory — a context manager / guidance helper that auto-injects
   relevant memories into the model's context at the start of each turn.

2. dakera_remember() — a @guidance function that stores a captured
   variable (or an explicit string) to Dakera after generation.

3. DakeraTools — pre-built guidance.Tool objects for store/search/forget
   so agents can call Dakera as a native tool inside gen(..., tools=[...]).

Quick start::

    from guidance import system, user, assistant, gen, models
    from guidance_dakera import DakeraMemory, dakera_remember

    lm = models.OpenAI("gpt-4o-mini")
    mem = DakeraMemory(agent_id="my-agent", base_url="http://localhost:3300")

    with system():
        lm += "You are a helpful assistant."

    with mem.inject(lm) as lm:        # injects top-k memories into system
        with user():
            lm += "What did we discuss last time?"
        with assistant():
            lm += gen(name="answer", max_tokens=200)
        lm = lm + dakera_remember("answer")   # persists the captured variable
"""

from ._client import DakeraClient
from ._memory import DakeraMemory, dakera_remember
from ._tools import DakeraTools

__all__ = [
    "DakeraClient",
    "DakeraMemory",
    "DakeraTools",
    "dakera_remember",
]
