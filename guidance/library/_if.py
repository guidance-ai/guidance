import re

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

    # see if we have a single block or multiple blocks TODO: this is a bit hacky, could be cleaned up with a better parsing I think...
    if len(block_content) > 1:
        if not (hasattr(block_content[1], "text") and re.match(r"{{~?\w*else\w*~?}}",  block_content[1].text) or re.search(r"${{~?\w*elif\w",  block_content[1].text)):
            block_content = [block_content]
    
    assert len(block_content) % 2 == 1, "Unexpected number of blocks for `if` command: " + str(len(block_content))

    # parse the first block
    if invert:
        value = not value
    if value:
        return await parser.visit(block_content[0], variable_stack)

    # parse the rest of the blocks
    for i in range(1, len(block_content), 2):

        # elif block
        if re.search(r"${{~?\w*elif\w",  block_content[i].text): # TODO: should probably depend on the parser not regex for this
            if parser.visit(block_content[i][1]):
                return await parser.visit(block_content[i+1], variable_stack)

        # else block
        elif re.match(r"{{~?\w*else\w*~?}}",  block_content[i].text):
            return await parser.visit(block_content[i+1], variable_stack)
        
        else:
            raise ValueError("Unexpected block content separator for `if` command: " + block_content[i].text)
    return ""
if_.is_block = True