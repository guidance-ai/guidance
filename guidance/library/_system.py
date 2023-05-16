from ._role import role

async def system(hidden=False, _parser_context=None):
    ''' A chat role block for the 'system' role.

    This is just a shorthand for {{#role 'system'}}...{{/role}}.

    Parameters
    ----------
    hidden : bool
        Whether to include the assistant block in future LLM context. 
    '''
    return await role(name="system", hidden=hidden, _parser_context=_parser_context)
system.is_block = True