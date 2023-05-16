def break_():
    ''' Breaks out of the current loop.

    This is useful for breaking out of a geneach loop early, typically this is used
    inside an `{{#if ...}}...{{/if}}` block.
    '''
    raise StopIteration()