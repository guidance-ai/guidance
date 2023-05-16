from ._role import role

async def assistant(hidden=False, _parser_context=None):
    ''' A chat role block for the 'assistant' role.

    This is just a shorthand for {{#role 'assistant'}}...{{/role}}.

    Parameters
    ----------
    hidden : bool
        Whether to include the assistant block in future LLM context. 
    '''
    return await role(name="assistant", hidden=hidden, _parser_context=_parser_context)
assistant.is_block = True