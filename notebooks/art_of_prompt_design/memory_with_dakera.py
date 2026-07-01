"""
guidance-dakera demo script
============================

Demonstrates all three integration styles:
  1. DakeraMemory.inject() — passive context injection
  2. dakera_remember()     — @guidance function for post-gen persistence
  3. DakeraTools.all()     — agentic tool calling (store / search / forget)

Requirements
------------
  pip install guidance guidance-dakera

  # Start Dakera (no API key needed for the demo):
  docker run -p 3300:3300 ghcr.io/dakera-ai/dakera:latest

  # Set your model backend.  Example with OpenAI:
  export OPENAI_API_KEY=sk-...
"""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
import guidance
from guidance import assistant, gen, models, system, user

from guidance_dakera import DakeraMemory, DakeraTools, dakera_remember

DAKERA_URL = "http://localhost:3300"
AGENT_ID = "demo-bot"

# Change to any backend you have available:
#   models.OpenAI("gpt-4o-mini")
#   models.LlamaCpp("/path/to/model.gguf")
#   models.Transformers("microsoft/Phi-4-mini-instruct")
try:
    lm = models.OpenAI("gpt-4o-mini")
    print("Using OpenAI gpt-4o-mini")
except Exception:
    from guidance.models import Mock
    lm = Mock(b"<s>I remember you mentioned neural scaling laws in our last chat.")
    print("OpenAI unavailable — using Mock model for demo")


# ---------------------------------------------------------------------------
# Style 1: DakeraMemory.inject() — automatic context injection
# ---------------------------------------------------------------------------
print("\n=== Style 1: Inject + Remember ===")

mem = DakeraMemory(
    agent_id=AGENT_ID,
    base_url=DAKERA_URL,
    top_k=3,
    min_score=0.0,
)

# Pre-seed a memory so the demo has something to retrieve
mem.remember(
    "The user is researching neural scaling laws, specifically Chinchilla paper.",
    tags=["research", "ml"],
    importance=0.8,
)

with system():
    lm += "You are a helpful research assistant."

# On the next user turn, inject relevant memories before the assistant replies
with mem.inject(lm, query="scaling laws research") as lm:
    with user():
        lm += "What should I read about LLM scaling?"
    with assistant():
        lm += gen(name="reply", max_tokens=150)

print(f"\nAssistant said:\n{lm['reply']}")

# Persist the assistant's reply back into memory
mem.remember(lm["reply"], tags=["assistant-response"])
print("Stored assistant reply to Dakera.")


# ---------------------------------------------------------------------------
# Style 2: dakera_remember() — inline @guidance function
# ---------------------------------------------------------------------------
print("\n=== Style 2: dakera_remember() ===")

with system():
    lm += "You are a concise summarizer."

with user():
    lm += "Summarize the Chinchilla scaling paper in one sentence."

with assistant():
    lm += gen(name="summary", max_tokens=80)

# Immediately after gen(), append dakera_remember to persist the captured var
lm = lm + dakera_remember(
    "summary",
    agent_id=AGENT_ID,
    base_url=DAKERA_URL,
    importance=0.9,
    tags=["summary", "chinchilla"],
)

print(f"\nSummary:\n{lm['summary']}")
print("Stored summary via dakera_remember().")


# ---------------------------------------------------------------------------
# Style 3: DakeraTools — agentic tool calling
# ---------------------------------------------------------------------------
print("\n=== Style 3: Agentic Tool Calling ===")

tools = DakeraTools(agent_id=AGENT_ID, base_url=DAKERA_URL, top_k=5)

with system():
    lm += (
        "You are a research assistant with persistent memory. "
        "Use dakera_search to look up what you already know before answering. "
        "Use dakera_store to save important new facts for later."
    )

with user():
    lm += "What do you know about the Chinchilla paper? And please remember that I prefer concise summaries."

with assistant():
    lm += gen(tools=tools.all(), max_tokens=500)

print(f"\nFull assistant turn:\n{str(lm)[-1000:]}")
