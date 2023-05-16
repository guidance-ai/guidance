import re

async def if_(value, invert=False, _parser_context=None):
    ''' Standard if/else statement.

    Parameters
    ----------
    value : bool
        The value to check. If `True` then the first block will be executed, otherwise the second block
        (the one after the `{{else}}`) will be executed.
    invert : bool
        If `True` then the value will be inverted before checking.
    '''
    block_content = _parser_context['block_content']
    assert len(block_content) in [1,3] # we don't support elseif yet...
    options = [block_content[0]]
    for i in range(1, len(block_content), 2):
        assert re.match(r"{{~?else~?}}",  block_content[i].text), "Expected else statement"
        options.append(block_content[i+1])

    # if isinstance(value, str):
    #     value2 = value
    #     value = value.lower().strip() in ["true", "yes", "on", "t", "y", "ok", "okay"]
    
    if invert:
        value = not value
    
    if value:
        return await _parser_context['parser'].visit(options[0])
    elif len(options) > 1:
        return await _parser_context['parser'].visit(options[1])
    else:
        return ""
if_.is_block = True