import pytest
import weakref
import gc
import inspect

import guidance
from guidance import gen, role


def test_dedent_basic():
    """Test that dedent functionality in f-strings works across Python versions."""

    @guidance(stateless=True, dedent=True)
    def character_maker(lm):
        lm += f"""\
        {{
            "name": "{1 + 1}",
            "age": "{gen("name", stop='"', max_tokens=1)}",
        }}"""
        return lm

    lm = guidance.models.Mock()
    result = lm + character_maker()
    assert str(result).startswith("{")


def test_basic_multiline_fstring():
    """Test a simple multiline f-string."""

    @guidance(stateless=True, dedent=True)
    def character_maker(lm):
        lm += f"""\
        {{
            "name": "{"har" + "sha"}",
            "age": "{314}",
        }}"""
        return lm

    lm = guidance.models.Mock()
    result = lm + character_maker()
    assert str(result) == '{\n    "name": "harsha",\n    "age": "314",\n}'


def test_mixed_content():
    """Test mixed f-strings and regular strings."""

    @guidance(stateless=True, dedent=True)
    def mixed_content(lm):
        s = "Regular string\n"
        s += f"""\
        {{
            "name": "{"har" + "sha"}",
            "age": "{314}",
        }}"""
        lm += s
        return lm

    lm = guidance.models.Mock()
    result = lm + mixed_content()
    assert str(result) == 'Regular string\n{\n    "name": "harsha",\n    "age": "314",\n}'


def test_non_fstring_multiline():
    """Test multiline strings that are not f-strings."""

    @guidance(stateless=True, dedent=True)
    def non_fstring_multiline(lm):
        s = """\
        Line 1
        Line 2
        Line 3
        """
        lm += s
        return lm

    lm = guidance.models.Mock()
    result = lm + non_fstring_multiline()
    assert str(result) == "Line 1\nLine 2\nLine 3\n"


def test_empty_strings():
    """Test empty strings."""

    @guidance(stateless=True, dedent=True)
    def empty_string(lm):
        s = f"""\
        {""}"""
        lm += s
        return lm

    lm = guidance.models.Mock()
    result = lm + empty_string()
    assert str(result) == ""


def test_inconsistent_indentation():
    """Test strings with inconsistent indentation."""

    @guidance(stateless=True, dedent=True)
    def inconsistent_indentation(lm):
        s = f"""\
        {{
        "name": "{"har" + "sha"}",
          "age": "{314}",
        "weapon": "{"sword"}"
        }}"""
        lm += s
        return lm

    lm = guidance.models.Mock()
    result = lm + inconsistent_indentation()
    assert str(result) == '{\n"name": "harsha",\n  "age": "314",\n"weapon": "sword"\n}'


# NOTE [HN]: The following two tests currently don't work, but they're fairly special/rare cases.
# Some implementation thoughts for the future:
# Nested f-strings: try creating a custom handler for ast.FormattedValue in the handler?
# closure functions: storing and rebinding vars referenced in func globals when recompiling?
# def test_nested_fstrings():
#     """Test nested f-strings."""
#     @guidance(stateless=True, dedent=True)
#     def nested_fstring(lm):
#         lm += f"""\
#         Outer {{
#             "inner": f"{{
#                 "value": {1+1}
#             }}"
#         }}
#         """
#         return lm

#     lm = guidance.models.Mock()
#     result = lm + nested_fstring()
#     assert str(result) == 'Outer {\n    "inner": "{\n    "value": 2\n}"\n}'

# def test_closure_function():
#     """Test function with closures referring to outer variables."""
#     @guidance(stateless=True, dedent=True)
#     def outer_function(lm):
#         outer_var = "outer_value"

#         def inner_function():
#             inner_var = f"""\
#             Inner function variable:
#                 outer_var: {outer_var}
#                 """
#             return inner_var
#         lm += inner_function()
#         return lm

#     lm = guidance.models.Mock()
#     result = lm + outer_function()
#     assert result == "Inner function variable:\nouter_var: outer_value\n"


def test_exception_on_repeat_calls():
    @guidance(stateless=True, dedent=False)
    def raises(lm):
        assert False

    with pytest.raises(AssertionError):
        raises()
    with pytest.raises(AssertionError):
        # Test against failure to reset the grammar function;
        # improper handling may not raise and may instead return
        # a Placeholder grammar node
        raises()


class TestGuidanceMethodCache:
    class MyClass:
        def __init__(self, prefix: str, suffix: str):
            self.prefix = prefix
            self.suffix = suffix
            self.delimiter = "\n"

        def __hash__(self):
            # Intentionally leave out self.delimiter so we can mess with it later
            return hash((self.prefix, self.suffix))

        @guidance(stateless=True, cache=True)
        def cached_method(self, lm, middle: str):
            return lm + self.delimiter.join([self.prefix, middle, self.suffix])

        @guidance(stateless=True, cache=False)
        def uncached_method(self, lm, middle: str):
            return lm + self.delimiter.join([self.prefix, middle, self.suffix])

    def test_guidance_method_cache(self):
        obj = self.MyClass("You are a helpful AI. Do what the user asks:", "Thank you.")
        grammar1 = obj.cached_method("Computer, tell me a joke.").value
        grammar2 = obj.cached_method("Computer, tell me a joke.").value
        assert grammar1 is grammar2

    def test_miss_cache_when_args_change(self):
        obj = self.MyClass("You are a helpful AI. Do what the user asks:", "Thank you.")
        grammar1 = obj.cached_method("Computer, tell me a joke.").value
        grammar2 = obj.cached_method("Computer, tell me a riddle.").value
        assert grammar1 is not grammar2
        lm = guidance.models.Mock()
        assert (
            str(lm + grammar1) == "You are a helpful AI. Do what the user asks:\nComputer, tell me a joke.\nThank you."
        )
        assert (
            str(lm + grammar2)
            == "You are a helpful AI. Do what the user asks:\nComputer, tell me a riddle.\nThank you."
        )

    def test_miss_cache_when_instance_hash_changes(self):
        obj = self.MyClass("You are a helpful AI. Do what the user asks:", "Thank you.")
        grammar1 = obj.cached_method("Computer, tell me a joke.").value
        obj.suffix = "Thanks!"
        grammar2 = obj.cached_method("Computer, tell me a joke.").value
        assert grammar1 is not grammar2
        lm = guidance.models.Mock()
        assert (
            str(lm + grammar1) == "You are a helpful AI. Do what the user asks:\nComputer, tell me a joke.\nThank you."
        )
        assert str(lm + grammar2) == "You are a helpful AI. Do what the user asks:\nComputer, tell me a joke.\nThanks!"

    def test_hit_cache_when_instance_hash_does_not_change(self):
        """
        Note: this is a bit of a "gotcha" when using `cache=True` since users may expect changing the instance's attributes
        will change the grammar. They _must_ implement __hash__ to ensure that the grammar is recalculated when the hash changes.
        """
        obj = self.MyClass("You are a helpful AI. Do what the user asks:", "Thank you.")
        grammar1 = obj.cached_method("Computer, tell me a joke.").value
        obj.delimiter = "\t"
        grammar2 = obj.cached_method("Computer, tell me a joke.").value
        assert grammar1 is grammar2
        lm = guidance.models.Mock()
        assert (
            str(lm + grammar1) == "You are a helpful AI. Do what the user asks:\nComputer, tell me a joke.\nThank you."
        )
        # Note that the delimiter is still the same as the first call since the hash didn't change
        assert (
            str(lm + grammar2) == "You are a helpful AI. Do what the user asks:\nComputer, tell me a joke.\nThank you."
        )

    def test_guidance_method_no_cache(self):
        obj = self.MyClass("You are a helpful AI. Do what the user asks:", "Thank you.")
        grammar1 = obj.uncached_method("Computer, tell me a joke.").value
        grammar2 = obj.uncached_method("Computer, tell me a joke.").value
        assert grammar1 is not grammar2
        lm = guidance.models.Mock()
        assert str(lm + grammar1) == str(lm + grammar2)

    def test_guidance_method_no_cache_changes_when_instance_changes(self):
        obj = self.MyClass("You are a helpful AI. Do what the user asks:", "Thank you.")
        grammar1 = obj.uncached_method("Computer, tell me a joke.").value
        obj.delimiter = "\t"
        grammar2 = obj.uncached_method("Computer, tell me a joke.").value
        assert grammar1 is not grammar2
        lm = guidance.models.Mock()
        assert (
            str(lm + grammar1) == "You are a helpful AI. Do what the user asks:\nComputer, tell me a joke.\nThank you."
        )
        # Note that the delimiter actually changes because the instance changed and we're not calling the cached method
        assert (
            str(lm + grammar2) == "You are a helpful AI. Do what the user asks:\tComputer, tell me a joke.\tThank you."
        )


class TestGuidanceMethodDedent:
    def test_dedent_basic(self):
        class MyClass:
            @guidance(stateless=True, dedent=True)
            def dedent_method(self, lm):
                lm += f"""\
                {{
                    "name": "{1 + 1}",
                    "age": "{gen("name", stop='"', max_tokens=1)}",
                }}"""
                return lm

        obj = MyClass()
        grammar = obj.dedent_method()
        lm = guidance.models.Mock()
        result = lm + grammar
        assert str(result).startswith("{")

    def test_basic_multiline_fstring(self):
        class MyClass:
            @guidance(stateless=True, dedent=True)
            def dedent_method(self, lm):
                lm += f"""\
                {{
                    "name": "{"har" + "sha"}",
                    "age": "{314}",
                }}"""
                return lm

        obj = MyClass()
        grammar = obj.dedent_method()
        lm = guidance.models.Mock()
        result = lm + grammar
        assert str(result) == '{\n    "name": "harsha",\n    "age": "314",\n}'

    def test_mixed_content(self):
        class MyClass:
            @guidance(stateless=True, dedent=True)
            def dedent_method(self, lm):
                s = "Regular string\n"
                s += f"""\
                {{
                    "name": "{"har" + "sha"}",
                    "age": "{314}",
                }}"""
                lm += s
                return lm

        obj = MyClass()
        grammar = obj.dedent_method()
        lm = guidance.models.Mock()
        result = lm + grammar
        assert str(result) == 'Regular string\n{\n    "name": "harsha",\n    "age": "314",\n}'

    def test_non_fstring_multiline(self):
        class MyClass:
            @guidance(stateless=True, dedent=True)
            def dedent_method(self, lm):
                s = """\
                Line 1
                Line 2
                Line 3
                """
                lm += s
                return lm

        obj = MyClass()
        grammar = obj.dedent_method()
        lm = guidance.models.Mock()
        result = lm + grammar
        assert str(result) == "Line 1\nLine 2\nLine 3\n"

    def test_empty_strings(self):
        class MyClass:
            @guidance(stateless=True, dedent=True)
            def dedent_method(self, lm):
                s = f"""\
                {""}"""
                lm += s
                return lm

        obj = MyClass()
        grammar = obj.dedent_method()
        lm = guidance.models.Mock()
        result = lm + grammar
        assert str(result) == ""

    def test_inconsistent_indentation(self):
        class MyClass:
            @guidance(stateless=True, dedent=True)
            def dedent_method(self, lm):
                s = f"""\
                {{
                "name": "{"har" + "sha"}",
                  "age": "{314}",
                "weapon": "{"sword"}"
                }}"""
                lm += s
                return lm

        obj = MyClass()
        grammar = obj.dedent_method()
        lm = guidance.models.Mock()
        result = lm + grammar
        assert str(result) == '{\n"name": "harsha",\n  "age": "314",\n"weapon": "sword"\n}'


class TestGuidanceRecursion:
    class MyClass:
        @guidance(stateless=True, dedent=False)
        def recursive(self, lm):
            return lm + guidance.select(["a", self.recursive()])

    def test_method_recursion(self):
        assert self.MyClass().recursive() is not None

    def test_function_recursion(self):
        @guidance(stateless=True, dedent=False)
        def recursive(lm):
            return lm + guidance.select(["a", recursive()])

        assert recursive() is not None


class TestMethodGarbageCollection:
    class MyClass:
        def __init__(self, thing: str = "Hello"):
            self.thing = thing

        def __hash__(self):
            return hash(self.thing)

        @guidance(stateless=True, cache=True)
        def cached_method(self, lm, *args):
            return lm + self.thing

        @guidance(stateless=True, cache=False)
        def uncached_method(self, lm, *args):
            return lm + self.thing

    def test_garbage_collection_cached_method(self):
        obj = self.MyClass()
        # Create a weak reference to the object
        obj_ref = weakref.ref(obj)
        # Call the cached method
        _ = obj.cached_method()
        # Delete the hard ref to the obj
        del obj
        # Run garbage collection
        gc.collect()
        # Check if the object was garbage collected
        assert obj_ref() is None

    def test_garbage_collection_uncached_method(self):
        obj = self.MyClass()
        # Create a weak reference to the object
        obj_ref = weakref.ref(obj)
        # Call the uncached method
        _ = obj.uncached_method()
        # Delete the hard ref to the obj
        del obj
        # Run garbage collection
        gc.collect()
        # Check if the object was garbage collected
        assert obj_ref() is None

    def test_deleting_instance_lets_method_be_garbage_collected(self):
        obj = self.MyClass()
        # Create a weak reference to the object
        obj_ref = weakref.ref(obj)
        # Create a weak reference to the cached method
        meth_ref = weakref.WeakMethod(obj.cached_method)
        # Quick sanity check that the weak reference is working
        gc.collect()
        assert meth_ref() is not None
        # Delete the hard ref to the obj
        del obj
        # Run garbage collection
        gc.collect()
        # Check if the object was garbage collected
        assert meth_ref() is None

    def test_deleting_instance_does_not_break_method(self):
        # Reference to method but not instance
        method = self.MyClass().cached_method
        gc.collect()
        # Will raise a ReferenceError if the method is broken
        method()


class TestSignature:
    def test_function_signature(self):
        def func(a, b=1, *, c, d=2):
            pass

        @guidance(stateless=True)
        def guidance_func(lm, a, b=1, *, c, d=2):
            return lm

        assert inspect.signature(guidance_func) == inspect.signature(func)

    def test_method_signature(self):
        class MyClass:
            def method(self, a, b=1, *, c, d=2):
                pass

            @guidance(stateless=True)
            def guidance_method(self, lm, a, b=1, *, c, d=2):
                pass

        obj = MyClass()
        assert inspect.signature(obj.guidance_method) == inspect.signature(obj.method)


def test_roles_in_stateless():
    """Test that roles are not allowed in stateless mode."""

    @guidance(stateless=True)
    def foo(lm):
        with role("assistant"):
            lm += gen()
        return lm

    with pytest.raises(RuntimeError, match="Cannot use roles or other blocks when stateless=True"):
        foo()
