from .._guidance import guidance
from ._block import block
from ._set_attribute import set_attribute


@guidance
def role_opener(lm, role_name, **kwargs):
    lm += lm.role_opener(role_name, **kwargs)
    return lm


@guidance
def role_closer(lm, role_name, **kwargs):
    lm += lm.role_closer(role_name, **kwargs)
    return lm


# TODO HN: Add a docstring to better describe arbitrary role functions
def role(role_name, text=None, **kwargs):
    if text is None:
        return block(
            name=role_name,
            opener=role_opener(role_name, **kwargs),
            closer=role_closer(role_name, **kwargs),
        )
    else:
        assert False
        # return self.append(open_text + text + close_text)


def system(text=None, **kwargs):
    """Indicate the 'system' prompt

    A convention has grown up around 'chat' APIs that
    prompts are split into three parts: system, user
    and assistant.
    This indicates the start of a 'system' block, which
    provides background information to the LLM.

        >>> with system():
        >>>     lm += "A system prompt"

    """
    return role("system", text, **kwargs)


def user(text=None, **kwargs):
    """Indicate the 'user' prompt

    A convention has grown up around 'chat' APIs that
    prompts are split into three parts: system, user
    and assistant.
    This indicates the start of a 'user' block, which
    provides input to the LLM from the user.

        >>> with user():
        >>>     lm += "What the user said"

    """
    return role("user", text, **kwargs)


def assistant(text=None, **kwargs):
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
    return role("assistant", text, **kwargs)


def function(text=None, **kwargs):
    return role("function", text, **kwargs)


def instruction(text=None, **kwargs):
    return role("instruction", text, **kwargs)


def indent_roles(indent=True):
    return set_attribute("indent_roles", indent)
