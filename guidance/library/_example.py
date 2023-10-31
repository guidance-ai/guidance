from ._role import role


async def example(hidden=False, _parser_context=None, **kwargs):
    """A chat role block for the 'example' role.

    This is just a shorthand for {{#role 'example'}}...{{/role}}.

    Parameters
    ----------
    hidden : bool
        Whether to include the block in future LLM context.
    """
    return await role(
        role_name="example", hidden=hidden, _parser_context=_parser_context, **kwargs
    )


example.is_block = True
