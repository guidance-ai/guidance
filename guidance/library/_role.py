from .._guidance import guidance
from ._block import block
from ._set_attribute import set_attribute

nodisp_start = "<||_#NODISP_||>"
nodisp_end = "<||_/NODISP_||>"
span_start = "<||_html:<span style='background-color: rgba(255, 180, 0, 0.3); border-radius: 3px;'>_||>"
span_end = "<||_html:</span>_||>"

@guidance
def role_opener(lm, role_name, **kwargs):
    indent = getattr(lm, "indent_roles", True)


    # Block start container (centers elements)
    if indent:
        lm += f"<||_html:<div style='display: flex; border-bottom: 1px solid rgba(127, 127, 127, 0.2);  justify-content: center; align-items: center;'><div style='flex: 0 0 80px; opacity: 0.5;'>{role_name.lower()}</div><div style='flex-grow: 1; padding: 5px; padding-top: 10px; padding-bottom: 10px; margin-top: 0px; white-space: pre-wrap; margin-bottom: 0px;'>_||>"

    # Start of either debug or HTML no disp block
    if indent:
        lm += nodisp_start
    else:
        lm += span_start
    
    # TODO [HN]: Temporary change while I instrument chat_template in transformers only.
    # Eventually have all models use chat_template.
    if hasattr(lm, "get_role_start"):
        lm += lm.get_role_start(role_name, **kwargs)
    elif hasattr(lm, "chat_template"):
        lm += lm.chat_template.get_role_start(role_name)
    else:
        raise Exception(
            f"You need to use a chat model in order the use role blocks like `with {role_name}():`! Perhaps you meant to use the {type(lm).__name__}Chat class?"
        )

    # End of either debug or HTML no disp block
    if indent:
        lm += nodisp_end
    else:
        lm += span_end

    return lm


@guidance
def role_closer(lm, role_name, **kwargs):
    indent = getattr(lm, "indent_roles", True)
    # Start of either debug or HTML no disp block
    if indent:
        lm += nodisp_start
    else:
        lm += span_start

    # TODO [HN]: Temporary change while I instrument chat_template in transformers only.
    # Eventually have all models use chat_template.
    if hasattr(lm, "get_role_end"):
        lm += lm.get_role_end(role_name)
    elif hasattr(lm, "chat_template"):
        lm += lm.chat_template.get_role_end(role_name)

    # End of either debug or HTML no disp block
    if indent:
        lm += nodisp_end
    else:
        lm += span_end

    # End of top container
    if indent:
        lm += "<||_html:</div></div>_||>"

    return lm

# TODO HN: Add a docstring to better describe arbitrary role functions
def role(role_name, text=None, **kwargs):
    if text is None:
        return block(
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
