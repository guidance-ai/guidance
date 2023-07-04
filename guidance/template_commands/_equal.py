def equal(*args):
    ''' Check that all arguments are equal.
    '''
    args[0]
    for arg in args[1:]:
        if arg != args[0]:
            return False
    return True