from ._role import role

async def assistant(hidden=False, _parser_context=None, **kwargs):
    ''' A chat role block for the 'assistant' role.

    This is just a shorthand for {{#role 'assistant'}}...{{/role}}.

    Parameters
    ----------
    hidden : bool
        Whether to include the assistant block in future LLM context. 
    '''
    return await role(role_name="assistant", hidden=hidden, _parser_context=_parser_context, **kwargs)
assistant.is_block = True