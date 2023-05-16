async def await_(name, _parser_context=None):
    ''' Awaits a variable by returning its value and then deleting it.

    Note that this is useful for repeatedly getting values since programs
    will pause when they need a value that is not yet set. This means
    that putting `await` in a loop will create a stateful "agent" that can
    repeatedly await values when called multiple times.

    Parameters
    ----------
    name : str
        The name of the variable to await.
    '''

    # stop the program completion if we are waiting for a value to be set
    # this will result in a partially completed program that we can then finish
    # later (by calling it again with the variable we need)
    parser = _parser_context['parser']
    if name not in parser.program:
        parser.executing = False
    else:
        value = parser.program[name]
        del parser.program[name]
        return value
    
    # cache = parser.program._await_cache
    # while name not in cache:
    #     parser.program.finish_execute() # allow the program to finish the current call (since we're waiting for a value from the next call now)
    #     # TODO: instead of waiting here, we should just single we are stopping the program completion here
    #     #       and then let all the containing elements record their state into a new program string that
    #     #       we can then use to continue the program completion later in a new object.
    #     cache.update(await parser.program._await_queue.get())
    #     pass
    # value = cache[name]
    # del cache[name]
    # return value