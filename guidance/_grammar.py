import base64
import uuid
import json
import inspect
import functools

tag_start = "{{G|"
tag_end = "|G}}"
_call_pool = {}
class StatefulException(Exception):
    '''This is raised when we try and use the state of a grammar object like it was a live model.
    
    Note that eventually we do want to support stateful parser/grammar constructs directly, but
    for now we use a traditional parser and grammar separation (hence the need for this exception).'''
    pass

_excluded_names = frozenset(["_find_name", "__add__", "__radd__", "<listcomp>", "<module>", "select", "char_set"])
def _find_name():
    stack = inspect.stack()
    for frame in stack:
        name = frame.function
        if frame.function not in _excluded_names:
            return name
    return None

class Function():
    def __init__(self, name, value=None) -> None:
        self.name = name
        self.value = value

    def __str__(self):
        '''Creates a string tag that can be used to retrieve this object.'''
    
        # save the call in our call pool, ready to be run when it is attached to an LM object
        str_id = str(id(self))
        if str_id not in _call_pool:
            _call_pool[str_id] = self

        # return a string representation of this call so it can be combined with other strings/calls
        return tag_start + str_id + tag_end

class StatefulFunction(Function):
    def __init__(self, f, args, kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, model):
        return self.f(model, *self.args, **self.kwargs)
    
    def __add__(self, other):
        if isinstance(other, str):
            other = _string(other)
        def __add__(model):
            model = self(model)
            if isinstance(other, StatelessFunction):
                return model + other
            else:
                return other(model)
        return StatefulFunction(__add__, [], {})
    
    def __radd__(self, other):
        if isinstance(other, str):
            other = _string(other)
        def __radd__(model):
            if isinstance(other, StatelessFunction):
                model += other
            else:
                model = other(model)
            return self(model)
        return StatefulFunction(__radd__, [], {})

class StatelessFunction(Function):
    used_names = set()

    def __add__(self, value):
        if isinstance(value, str) or isinstance(value, bytes):
            value = _string(value)
        return Join([self, value], name=_find_name() + "_" + StatelessFunction._new_name())
    
    def __radd__(self, value):
        if isinstance(value, str) or isinstance(value, bytes):
            value = _string(value)
        return Join([value, self], name=_find_name() + "_" + StatelessFunction._new_name())
    
    def __getitem__(self, value):
        raise StatefulException("StatelessFunctions can't access state!")
    
    @staticmethod
    def _new_name(used_names=None):
        if used_names is None:
            used_names = StatelessFunction.used_names

        # look for a name with one letter
        for c in range(ord('a'), ord('z')+1):
            name = chr(c)
            if name not in used_names:
                used_names.add(name)
                return name
        
        # if those are all used look for 2 letter names
        for c in range(ord('a'), ord('z')+1):
            for d in range(ord('a'), ord('z')+1):
                name = chr(c) + chr(d)
                if name not in used_names:
                    used_names.add(name)
                    return name
                    
        # if those are all used look for 3 letter names
        for c in range(ord('a'), ord('z')+1):
            for d in range(ord('a'), ord('z')+1):
                for e in range(ord('a'), ord('z')+1):
                    name = chr(c) + chr(d) + chr(e)
                    if name not in used_names:
                        used_names.add(name)
                        return name
        
        # fall back to a uuid name
        name = "a" + uuid.uuid4()
        used_names.add(name)
        return name
    
    def gbnf_string(self):
        used_names = set()
        names = {}
        lines = []
        root_name = self._rec_gbnf_string(lines, used_names, names)
        lines.append("root ::= " + root_name)
        return "\n".join(lines)

class Terminal(StatelessFunction):
    def match_byte(self, byte):
        pass # abstract

class Byte(Terminal):
    def __init__(self, byte):
        assert isinstance(byte, bytes)
        assert len(byte) == 1
        self.byte = byte
        self.hidden = False
        self.commit_point = False
        self.capture_name = None

    @property
    def name(self):
        return str(self.byte)
    
    def __hash__(self):
        return self.byte[0]
    
    def __eq__(self, other):
        return isinstance(other, Byte) and self.byte[0] == other.byte[0]
    
    def __repr__(self) -> str:
        return str(self.byte)
    
    def match_byte(self, byte):
        return byte == self.byte
    
    @property
    def nullable(self):
        return False

class ByteRange(Terminal):
    def __init__(self, byte_range):
        assert isinstance(byte_range, bytes)
        assert len(byte_range) == 2
        self.byte_range = byte_range
        self.hidden = False
        self.commit_point = False
        self.capture_name = None

    def match_byte(self, byte):
        return self.byte_range[0] <= byte[0] <= self.byte_range[1]

    @property
    def name(self):
        return str(self.byte_range)
    @name.setter
    def name(self, value):
        pass # we ignore name changes
    
    @property
    def nullable(self):
        return False
    
    def __hash__(self):
        return self.byte_range[0] + 256 * self.byte_range[1]
    
    def __eq__(self, other):
        return isinstance(other, ByteRange) and self.byte_range[0] == other.byte_range[0] and self.byte_range[1] == other.byte_range[1]
    
    def __repr__(self) -> str:
        return str(self.byte)

class Null():
    nullable = True
    def __init__(self):
        self.hidden = False
        self.commit_point = False
        self.capture_name = None

    def __add__(self, other):
        if isinstance(other, str):
            return _string(other)
        else:
            return other

class Join(StatelessFunction):
    def __init__(self, values, name=None) -> None:
        self.nullable = all(v.nullable for v in values)
        self.values = [v for v in values if not isinstance(v, Null)]
        self.name = name if name is not None else StatelessFunction._new_name()
        self.hidden = False
        self.commit_point = False
        self.capture_name = None

    def __repr__(self, indent="", done=None):
        if done is None:
            done = set()
        s = self.name + " <- " + " ".join([v.name for v in self.values]) + "\n"
        done.add(self)
        for v in self.values:
            if v not in done and (isinstance(v, Join) or isinstance(v, Select)):
                s += v.__repr__(indent, done)
        return s

class Select(StatelessFunction):
    def __init__(self, values, name=None) -> None:
        self.values = values
        self.name = name if name is not None else StatelessFunction._new_name()
        self.hidden = False
        self.commit_point = False
        self.capture_name = None

    @property
    def values(self):
        return self._values
    @values.setter
    def values(self, vals):
        self._values = [_string(v) if isinstance(v, str) else v for v in vals]
        self.nullable = any(v.nullable for v in self._values)
        self._values = [v for v in self._values if not isinstance(v, Null)]

    def __repr__(self, indent="", done=None):
        if done is None:
            done = set()
        s = self.name + " <- " + " | ".join([v.name for v in self.values]) + "\n"
        done.add(self)
        for v in self.values:
            if v not in done and (isinstance(v, Join) or isinstance(v, Select)):
                s += v.__repr__(indent, done)
        return s

def _string(value):
    if isinstance(value, str):
        b = bytes(value, encoding="utf8")
    elif isinstance(value, bytes):
        b = value
    else:
        raise Exception("Must pass bytes or str to the string() function!")
    if len(value) == 0:
        return Null()
    elif len(b) == 1:
        return Byte(b)
    else:
        return Join([Byte(b[i:i+1]) for i in range(len(b))], name='"' + str(value) + '"')
    
def _select(values, name=None, recurse=False):
    for value in values:
        assert not isinstance(value, StatefulFunction), "You cannot select between stateful functions in the current guidance implementation!"
    if name is None:
        name = _find_name() + "_" + StatelessFunction._new_name()
    if recurse:
        node = Select([], name)
        node.values = [v + node for v in values if v != ""] + values
        return node
    else:
        if len(values) == 1 and name is None:
            return values[0]
        else:
            return Select(values, name)
        
def _byte_range(byte_range):
    return ByteRange(byte_range)

def _capture(value, name=None):
    value.capture_name = name
    return value

# def char_range(low, high):
#     low_bytes = bytes(low, encoding="utf8")
#     high_bytes = bytes(high, encoding="utf8")
#     if len(low_bytes) > 1 or len(high_bytes) > 1:
#         raise Exception("We don't yet support multi-byte character ranges!")
#     return ByteRange(low_bytes + high_bytes)