async def role(role, block_content, partial_output, parser, next_node, prev_node, next_next_node, hidden=False):
    ''' A chat role block.
    '''
    assert parser is not None
    
    # record where we are in the prefix in case we need to rewind
    pos = len(parser.prefix)
    
    # send the role-start special tokens
    partial_output(parser.program.llm.role_start(role))

    out = await parser.visit(block_content[0], next_node=next_node, prev_node=prev_node, next_next_node=next_next_node)

    # send the role-end special tokens
    partial_output(parser.program.llm.role_end(role))
    
    # hide everything if needed
    if hidden:
        new_content = parser.prefix[pos:]
        parser.reset_prefix(pos)
        parser.extend_prefix("{{!--GHIDDEN:"+new_content.replace("--}}", "--_END_END")+"--}}")
    
    return out
role.is_block = True