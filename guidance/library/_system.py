from ._role import role

async def system(block_content, partial_output, parser, prev_node, next_node, next_next_node, hidden=False):
    ''' A chat role block for the 'system' role.

    This is just a shorthand for {{#role 'system'}}...{{/role}}.
    '''
    return await role(role="system", block_content=block_content, partial_output=partial_output, parser=parser, prev_node=prev_node, next_node=next_node, next_next_node=next_next_node, hidden=hidden)
system.is_block = True