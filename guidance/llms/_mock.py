from ._llm import LLM

class Mock(LLM):
    """ Mock class for testing.
    """
    def __init__(self, output=None):
        """ Initialize the mock class.
        
        Parameters
        ----------
        output : object
            The output of the mock class. If a list is provided, the mock
            class will return the next item in the list each time it is
            called. Otherwise, the same output will be returned each time.
        """
        if output is None:
            output = [f"mock output {i}" for i in range(100)]
        self.output = output
        self.index = 0

    def __call__(self, *args, n=1, stream=False, **kwds):
        choices = []
        for i in range(n):
            if isinstance(self.output, list):
                out = self.output[self.index]
                self.index += 1
            else:
                out = self.output

            choices.append({"text": out})

        out = {"choices": choices}

        if stream:
            return [out]
        else:
            return out
        
    def role_start(self, role):
        return "<|im_start|>"+role+"\n"
    
    def role_end(self, role=None):
        return "<|im_end|>"