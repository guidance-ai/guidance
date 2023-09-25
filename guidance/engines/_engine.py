class Engine():
    '''Base class for engines.'''
    
    def __call__(self, prompt, pattern, max_tokens=100, n=1, top_p=1, temperature=0.0, ensure_bos_token=True):
        pass # meant to be overriden by subclasses