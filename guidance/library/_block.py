from .._utils import strip_markers

async def block(name=None, hidden=False, _parser_context=None):
    ''' Generic block-level element.

    This is useful for naming or hiding blocks of content.

    Parameters
    ----------
    name : str
        The name of the block. A variable with this name will be set with the generated block content.
    hidden : bool
        Whether to include the generated block content in future LLM context.
    '''

    parser = _parser_context['parser']
    pos = len(parser.prefix)
    out = await parser.visit(
        _parser_context['block_content'][0],
        next_node=_parser_context["next_node"],
        next_next_node=_parser_context["next_next_node"],
        prev_node=_parser_context["prev_node"]
    )
    if name is not None:
        parser.set_variable(name, strip_markers(out))
    if hidden:
        new_content = parser.prefix[pos:]
        parser.reset_prefix(pos)
        parser.extend_prefix("{{!--GHIDDEN:"+new_content.replace("--}}", "--_END_END")+"--}}")
    
    return out
block.is_block = True