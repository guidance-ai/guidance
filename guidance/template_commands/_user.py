from ._role import role

async def user(hidden=False, _parser_context=None, **kwargs):
    ''' A chat role block for the 'user' role.

    This is just a shorthand for {{#role 'user'}}...{{/role}}.

    Parameters
    ----------
    hidden : bool
        Whether to include the assistant block in future LLM context. 
    '''
    return await role(role_name="user", hidden=hidden, _parser_context=_parser_context, **kwargs)
user.is_block = True