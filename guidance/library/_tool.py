
# @guidance
# def default_text_to_callable(lm, generated_text):

# class Tool:
#     def __init__(self, callable, scan_pattern=None, text_to_callable=None):
#         # callable is a guidance function
#         # Scan pattern is a pattern used to stop generation
#         # text_to_callable returns either None if the tool does not apply to the generated text, or calls 'callable'
#         # and executes the tool (whatever the tool does)
#         self.scan_pattern = scan_patter
#         self.text_to_callable = text_to_callable
#     def __call__(self, lm, text):




# Leaving this here for now in case we want to return to this pattern, but implemented a different version above for prototyping
class CallScanner:
    def __init__(self, scanner, stop=None, stop_regex=None):
        self.scanner = scanner
        self.stop = stop
        self.stop_regex = stop_regex
        assert self.stop is not None or self.stop_regex is not None, "Either stop or stop_regex must be specified."

    def __call__(self, lm, text):
        out = self.scanner(text)
        if isinstance(out, CallableAnswer) and out.callable is None:
            out.callable = lm.get(out.__name__, {"callable": None}).get("callable", None)
        return out

    
class CallableAnswer:
    def __init__(self, text, name, args_string, callable=None):
        self._text = text
        self.__name__ = name
        self.args_string = args_string
        self.callable = callable

    def __str__(self):
        return self._text

    def __call__(self, *args, **kwargs):
        if self.callable is None:
            raise NotImplementedError(f"Answer {self.__name__} has no function defined")
        return self.callable(*args, **self.__kwdefaults__, **kwargs)
    
    @property
    def __kwdefaults__(self):
        """We build this lazily in case the user wants to handle validation errors themselves."""
        return json.loads(self.args_string)

    def __repr__(self):
        return self._text + f"\nCallableAnswer(__name__={self.__name__}, __kwdefaults__={self.__kwdefaults__})"

def _default_extract_function_call(text):
    m = re.match(r"(.*?)\n?\n?```typescript\nfunctions.([^\(]+)\((.*?)\)```", text, re.DOTALL)
    if m:
        return CallableAnswer(text=m.group(1), name=m.group(2), args_string=m.group(3))
# _default_call_scanner = CallScanner(_extract_function_call, stop_regex=r"\n?\n?```typescript\nfunctions.[^\(]+\(.*?\)```")

# Old functions, to clean up
    # def add_call_scanner(self, scanner, stop=None, stop_regex=None):
    #     self._call_scanners.append(CallScanner(scanner, stop=stop, stop_regex=stop_regex))
    #     return self
    
    # def get_call_scanners(self):
    #     return self._call_scanners
