from .._utils import ContentCapture
from .._grammar import grammar

async def parse(string, name=None, hidden=False, _parser_context=None):
    ''' Parse a string as a guidance program.

    This is useful for dynamically generating and then running guidance programs (or parts of programs).

    Parameters
    ----------
    string : str
        The string to parse.
    name : str
        The name of the variable to set with the generated content.
    '''

    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']

    # capture the content of the block
    with ContentCapture(variable_stack, hidden) as new_content:

        # parse and visit the given string
        subtree = grammar.parse_string(string)
        new_content += await parser.visit(subtree, variable_stack)

        # save the content in a variable if needed
        if name is not None:
            variable_stack[name] = str(new_content)
