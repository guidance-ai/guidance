from .._utils import ContentCapture

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
    variable_stack = _parser_context['variable_stack']
    
    # capture the content of the block
    with ContentCapture(variable_stack, hidden) as new_content:

        # visit the block content
        new_content += await parser.visit(
            _parser_context['block_content'][0],
            variable_stack,
            next_node=_parser_context["next_node"],
            next_next_node=_parser_context["next_next_node"],
            prev_node=_parser_context["prev_node"]
        )

        # set the variable if needed
        if name is not None:
            variable_value = str(new_content)
            variable_stack[name] = variable_value
block.is_block = True