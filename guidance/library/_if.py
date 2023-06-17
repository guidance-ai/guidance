
async def if_(value, *, invert=False, _parser_context=None):
    ''' Standard if/else statement.

    Parameters
    ----------
    value : bool
        The value to check. If `True` then the first block will be executed, otherwise the second block
        (the one after the `{{else}}`) will be executed.
    invert : bool [DEPRECATED]
        If `True` then the value will be inverted before checking.
    '''
    block_content = _parser_context['block_content']
    variable_stack = _parser_context['variable_stack']
    parser = _parser_context['parser']
    
    assert len(block_content) % 2 == 1, "Unexpected number of blocks for `if` command: " + str(len(block_content))

    # parse the first block
    if invert:
        value = not value
    if value:
        return await parser.visit(block_content[0], variable_stack)

    # parse the rest of the blocks
    for i in range(1, len(block_content), 2):

        # elif block
        if block_content[i][0] == "elif":
            if parser.visit(block_content[i][1], variable_stack):
                return await parser.visit(block_content[i+1], variable_stack)

        # else block
        elif block_content[i][0] == "else":
            return await parser.visit(block_content[i+1], variable_stack)
        
        else:
            raise ValueError("Unexpected block content separator for `if` command: " + block_content[i].text)
    return ""
if_.is_block = True