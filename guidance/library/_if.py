async def if_(value, block_content, parser, reverse=False):
    ''' Standard if/else statement.
    '''
    assert len(block_content) in [1,3] # we don't support elseif yet...
    options = [block_content[0]]
    for i in range(1, len(block_content), 2):
        assert block_content[i].text == "{{else}}"
        options.append(block_content[i+1])

    # if isinstance(value, str):
    #     value2 = value
    #     value = value.lower().strip() in ["true", "yes", "on", "t", "y", "ok", "okay"]
    
    if reverse:
        value = not value
    
    if value:
        return await parser.visit(options[0])
    elif len(options) > 1:
        return await parser.visit(options[1])
    else:
        return ""
if_.is_block = True