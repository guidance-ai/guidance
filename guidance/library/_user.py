from ._role import role

async def user(block_content, partial_output, parser, prev_node, next_node, next_next_node, hidden=False):
    ''' A chat role block for the 'user' role.

    This is just a shorthand for {{#role 'user'}}...{{/role}}.
    '''
    return await role(role="user", block_content=block_content, partial_output=partial_output, parser=parser, prev_node=prev_node, next_node=next_node, next_next_node=next_next_node, hidden=hidden)
user.is_block = True