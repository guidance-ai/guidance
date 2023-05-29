from .._utils import strip_markers
from .._grammar import grammar

async def parse(string, name=None, _parser_context=None):
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
    pos = len(parser.prefix)
    # parse the string
    subtree = grammar.parse(string)
    out = await parser.visit(subtree)

    if name is not None:
        new_content = parser.prefix[pos:]
        parser.set_variable(name, strip_markers(new_content))

    return out
