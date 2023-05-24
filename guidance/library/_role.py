async def role(name, hidden=False, _parser_context=None):
    ''' A chat role block.
    '''
    block_content = _parser_context['block_content']
    parser = _parser_context['parser']
    partial_output = _parser_context['partial_output']
    
    # record where we are in the prefix in case we need to rewind
    pos = len(parser.prefix)
    
    # send the role-start special tokens
    partial_output(parser.program.llm.role_start(name))

    out = await parser.visit(
        block_content[0],
        next_node=_parser_context["block_close_node"],
        prev_node=_parser_context["prev_node"],
        next_next_node=_parser_context["next_node"]
    )

    # send the role-end special tokens
    partial_output(parser.program.llm.role_end(name))
    
    # hide everything if needed
    if hidden:
        new_content = parser.prefix[pos:]
        parser.reset_prefix(pos)
        parser.extend_prefix("{{!--GHIDDEN:"+new_content.replace("--}}", "--_END_END")+"--}}")
    
    return out
role.is_block = True