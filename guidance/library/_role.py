from contextlib import AbstractContextManager
from ..models.base import role


def system() -> AbstractContextManager[None]:
    """Indicate the 'system' prompt

    A convention has grown up around 'chat' APIs that
    prompts are split into three parts: system, user
    and assistant.
    This indicates the start of a 'system' block, which
    provides background information to the LLM.

        >>> with system():
        >>>     lm += "A system prompt"

    """
    return role("system")


def user() -> AbstractContextManager[None]:
    """Indicate the 'user' prompt

    A convention has grown up around 'chat' APIs that
    prompts are split into three parts: system, user
    and assistant.
    This indicates the start of a 'user' block, which
    provides input to the LLM from the user.

        >>> with user():
        >>>     lm += "What the user said"

    """
    return role("user")


def assistant() -> AbstractContextManager[None]:
    """Indicate the 'assistant' prompt

    A convention has grown up around 'chat' APIs that
    prompts are split into three parts: system, user
    and assistant.
    This indicates the start of an 'assistant' block, which
    marks LLM response (or where the LLM will generate
    the next response).

        >>> with assistant():
        >>>     lm += gen(name="model_output", max_tokens=20)

    """
    return role("assistant")
