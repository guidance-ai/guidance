from ._role import role

async def function(hidden=False, _parser_context=None, **kwargs):
    ''' A chat role block for the 'function' role.

    This is just a shorthand for {{#role 'function'}}...{{/role}}.

    Parameters
    ----------
    hidden : bool
        Whether to include the assistant block in future LLM context. 
    '''
    return await role(role_name="function", hidden=hidden, _parser_context=_parser_context, **kwargs)
function.is_block = True