from ._llm import LLM

class Mock(LLM):
    """ Mock class for testing.
    """

    def __init__(self, output=None):
        """ Initialize the mock class.
        
        Parameters
        ----------
        output : str or list or dict
            The output of the mock class. If a list is provided, the mock
            class will return the next item in the list each time it is
            called. Otherwise, the same output will be returned each time.
            If a dictionary is provided, the mock class will choose the first
            dictionary key that matches a suffix of the input prompt, and use
            the string or list value associated with that key for generation.
        """

        # make sure the output is always a dictionary of lists
        if output is None:
            output = {"": [f"mock output {i}" for i in range(100)]}
        if isinstance(output, str):
            output = [output]
        if isinstance(output, list):
            output = {"": output}
        for key in output.keys():
            if not isinstance(output[key], list):
                output[key] = [output[key]]
        
        self.output = output
        self.counts = {k: 0 for k in output.keys()}
        self._tokenizer = MockTokenizer()

    def _find_suffix_match(self, prompt):
        """ Find the output key that matches the suffix of the prompt.
        """

        for key in self.output.keys():
            if prompt.endswith(key):
                return key

    def __call__(self, prompt, *args, n=1, stream=False, **kwargs):
        key = self._find_suffix_match(prompt)
        output = self.output[key]
        choices = []
        for i in range(n):
            out = output[min(self.counts[key], len(output)-1)]
            self.counts[key] += 1
            if isinstance(out, str):
                choices.append({"text": out, "finish_reason": "stop"})
            elif isinstance(out, dict):
                choices.append(out)
            else:
                raise ValueError("Invalid output type: " + str(type(out)))

        out = {"choices": choices}

        if stream:
            return [out]
        else:
            return out
        
    def role_start(self, role):
        return "<|im_start|>"+role+"\n"
    
    def role_end(self, role=None):
        return "<|im_end|>"
    
class MockTokenizer():
    def __init__(self):
        pass

    def encode(self, text):
        return [s for s in text.encode("utf-8")]
    
    def decode(self, ids):
        return "".join([chr(i) for i in ids])