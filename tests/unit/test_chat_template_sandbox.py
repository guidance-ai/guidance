"""Tests for chat template rendering sandbox.

Chat templates are loaded from model files (HuggingFace Hub, local paths) and
may be attacker-controlled. The rendering must use a sandboxed Jinja2
environment to prevent server-side template injection (SSTI) → arbitrary code
execution via poisoned model files.

See: https://github.com/guidance-ai/guidance/pull/<TBD>
"""

import pytest
from jinja2 import BaseLoader
from jinja2.sandbox import ImmutableSandboxedEnvironment


def test_sandbox_blocks_attribute_access_ssti():
    """A malicious chat template that chains __init__.__globals__.__builtins__
    to reach os.popen must be blocked by the sandboxed environment."""
    malicious_template = "{{ self.__init__.__globals__.__builtins__.__import__('os').popen('id').read() }}"
    env = ImmutableSandboxedEnvironment(loader=BaseLoader)
    rtemplate = env.from_string(malicious_template)
    with pytest.raises(Exception):
        rtemplate.render(messages=[], tools=None, add_generation_prompt=True)


def test_sandbox_blocks_class_attribute_access():
    """The classic Jinja SSTI via __class__ chaining must be blocked."""
    malicious_template = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
    env = ImmutableSandboxedEnvironment(loader=BaseLoader)
    rtemplate = env.from_string(malicious_template)
    with pytest.raises(Exception):
        rtemplate.render(messages=[], tools=None, add_generation_prompt=True)


def test_sandbox_renders_normal_chat_template():
    """Normal chat templates (loops, conditionals, variable access) must render
    correctly under the sandbox — no false positives."""
    normal_template = "{% for message in messages %}<|{{ message.role }}|>{{ message.content }}{% endfor %}"
    env = ImmutableSandboxedEnvironment(loader=BaseLoader)
    rtemplate = env.from_string(normal_template)
    result = rtemplate.render(
        messages=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
        tools=None,
        add_generation_prompt=True,
    )
    assert "<|user|>Hello" in result
    assert "<|assistant|>Hi" in result


def test_sandbox_renders_template_with_tojson_filter():
    """The tojson filter (used in many real chat templates for tool
    definitions) must work under the sandbox."""
    template = "{{ tools | tojson }}"
    env = ImmutableSandboxedEnvironment(loader=BaseLoader)
    rtemplate = env.from_string(template)
    result = rtemplate.render(messages=[], tools=[{"name": "calc", "params": {"x": 1}}], add_generation_prompt=True)
    assert "calc" in result


def test_sandbox_renders_llama3_style_template():
    """A simplified Llama 3-style template with conditionals, loops, and
    variable interpolation must render correctly (the most common real-world
    template shape)."""
    template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ '<|start_header_id|>' + message.role + '<|end_header_id|>\\n\\n' }}"
        "{{ message.content | trim + '<|eot_id|>' }}"
        "{% endfor %}"
    )
    env = ImmutableSandboxedEnvironment(loader=BaseLoader)
    rtemplate = env.from_string(template)
    result = rtemplate.render(
        messages=[{"role": "user", "content": "  Hello  "}],
        tools=None,
        add_generation_prompt=True,
        bos_token="<|begin_of_text|>",
    )
    assert "<|begin_of_text|>" in result
    assert "<|start_header_id|>user<|end_header_id|>" in result
    assert "Hello<|eot_id|>" in result  # trim removes leading/trailing spaces
