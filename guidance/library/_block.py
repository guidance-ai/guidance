from .._utils import strip_markers

async def block(name=None, block_content=None, parser=None, hidden=False):
    ''' Generic block definition.
    '''
    assert parser is not None
    
    pos = len(parser.prefix)
    out = await parser.visit(block_content[0])
    if name is not None:
        parser.set_variable(name, strip_markers(out))
    if hidden:
        new_content = parser.prefix[pos:]
        parser.reset_prefix(pos)
        parser.extend_prefix("{{!--GHIDDEN:"+new_content.replace("--}}", "--_END_END")+"--}}")
    
    return out