from ._if import if_

async def unless(value, _parser_context=None):
    return await if_(value, invert=True, _parser_context=_parser_context)
unless.is_block = True