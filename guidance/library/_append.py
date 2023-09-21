import guidance

@guidance
def append(lm, content):
    '''This is basically a wrapper for the addition operation.
    
    This is useful because it is a guidance function and so supports streaming, async, etc.
    '''
    return lm + content