from typing import Any
from IPython.display import clear_output, display, HTML
import html
import re
import copy
import json
import textwrap

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

class LM:
    def __init__(self, model, caching=True, call_scanners=None):
        self.model = model
        self._state = ""
        self._children = []
        self._event_queue = None
        self._event_parent = None
        self._silent = None
        # self._inplace = None
        self._variables = {}
        self._caching = caching
        self._endpoint_session = None
        self.endpoint = None
        self.instance__enter__ = []
        self.instance__exit__ = []
        self._call_scanners = call_scanners
        if self._call_scanners is None:
            self._call_scanners = []
            self.add_call_scanner(_default_extract_function_call, stop_regex=r"\n?\n?```typescript\nfunctions.[^\(]+\(.*?\)```")

    def get(self, key, default=None):
        return self._variables.get(key, default)
    
    def remove(self, key):
        if key in self._variables:
            del self._variables[key]

    def add_call_scanner(self, scanner, stop=None, stop_regex=None):
        self._call_scanners.append(CallScanner(scanner, stop=stop, stop_regex=stop_regex))
        return self
    
    def get_call_scanners(self):
        return self._call_scanners

    def get_endpoint_session(self):
        return self._endpoint_session_call
    
    def _endpoint_session_call(self, *args, **kwargs):
        kwargs["caching"] = self._caching
        return self._endpoint_session(*args, **kwargs)

    def _html(self):
        display_out = html.escape(self._state)
        display_out = re.sub(r"&lt;\|\|_#NODISP_\|\|&gt;.*?&lt;\|\|_/NODISP_\|\|&gt;", "", display_out, flags=re.DOTALL)
        display_out = re.sub(r"&lt;\|\|_html:(.*?)_\|\|&gt;", lambda x: html.unescape(x.group(1)), display_out, flags=re.DOTALL)
        display_out = "<pre style='margin: 0px; padding: 0px; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'>"+display_out+"</pre>"
        return display_out
    
    def _send_to_event_queue(self, value):
        if self._event_queue is not None:
            self._event_queue.put(value)
        if self._event_parent is not None:
            self._event_parent._send_to_event_queue(value)

    @property
    def silent(self):
        if self._silent is not None:
            return self._silent
        return False
    
    def copy(self):
        new_lm = copy.copy(self)
        new_lm._event_queue = None
        if self._event_queue is not None:
            new_lm._event_parent = self
        new_lm._variables = self._variables.copy()
        new_lm._children = []
        self._children.append(new_lm)
        return new_lm
    
    def append(self, value, force_silent=False, dedent=True):
        """This is the base way to add content to the LM object."""
        if dedent and ('\n' in value or '\r' in value):
            value = textwrap.dedent(value)
        self._state += str(value)
        if not self.silent and not force_silent:
            clear_output(wait=True)
            display(HTML(self._html()))
        self._send_to_event_queue(self)
        return self
    
    def reset(self, position=0, clear_variables=True):
        """This resets the state of the LM prompt past the given postion."""
        self._state = self._state[:position]
        if clear_variables:
            self._variables = {}

    def _repr_html_(self):
        clear_output(wait=True)
        return self._html()
    
    def __str__(self) -> str:
        return re.sub(r"<\|\|_.*?_\|\|>", "", self._state)
    
    def __iadd__(self, value):
        return self.append(value)
    
    def __len__(self):
        return len(str(self))
    
    def __call__(self, s, **kwargs):
        return self.append(s, **kwargs)
    
    def __setitem__(self, key, value):
        self._variables[key] = value

    def __getitem__(self, key):
        return self._variables[key]
    
    def __contains__(self, item):
        return item in self._variables

    def __enter__(self):
        if len(self.instance__enter__) > 0:
            return self.instance__enter__.pop()()

    def __exit__(self, exc_type, exc_value, traceback):
        if len(self.instance__exit__) > 0:
            return self.instance__exit__.pop()(exc_type, exc_value, traceback)

    # def __call__(self, s):
    #     return self + s
    
    def get_encoded(self, s):
        return self.endpoint.encode(s)
    
    def get_decoded(self, s):
        return self.endpoint.decode(s)
    
    def get_id_to_token(self, id):
        return self.get_decoded([id])

    def get_token_to_id(self, token):
        return self.get_encoded(token)[0]
    
    def get_cache(self):
        return self.endpoint.cache
    
    def tool_def(self, functions):

        self += """
# Tools

"""
        if len(functions) > 0:
            self += '''## functions

namespace functions {

'''
        for function in functions:
            self += f"""// {function['description']}
type {function['name']} = (_: {{"""
            for prop_name,prop_data in function["parameters"]["properties"].items():
                if "description" in prop_data:
                    self += f"\n// {prop_data['description']}\n"
                self += prop_name
                if prop_name not in function["parameters"]["required"]:
                    self += "?"
                self += ": "
                if "enum" in prop_data:
                    for enum in prop_data["enum"]:
                        self += f'"{enum}"'
                        if enum != prop_data["enum"][-1]:
                            self += " | "
                else:
                    self += prop_data["type"]
                
                if prop_name != list(function["parameters"]["properties"].keys())[-1]:
                    self += ",\n"
            self += """
}) => any;

"""
            self[function['name']] = function
        self += "} // namespace functions\n"
        
        return self


class ChatLM(LM):
    
    def get_role_start(self, role_name, **kwargs):
        return "<|im_start|>"+role_name+"".join([f' {k}="{v}"' for k,v in kwargs.items()])+"\n"
    
    def get_role_end(self, role_name=None):
        return "<|im_end|>"