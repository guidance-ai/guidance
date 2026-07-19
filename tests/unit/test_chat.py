"""Offline tests for chat template registration."""

from jinja2 import BaseLoader, Environment

from guidance.chat import CHAT_TEMPLATE_CACHE, Qwen3ChatTemplate, qwen3_it_template


def _render(template_str, messages):
    return (
        Environment(loader=BaseLoader())
        .from_string(template_str)
        .render(add_generation_prompt=True, messages=messages, bos_token="", eos_token="")
    )


def test_qwen3_template_str_matches_qwen3():
    assert CHAT_TEMPLATE_CACHE[qwen3_it_template] is Qwen3ChatTemplate
    assert Qwen3ChatTemplate.template_str is qwen3_it_template


def test_qwen3_render_has_no_spurious_default_system():
    messages = [{"role": "user", "content": "Hi"}]
    rendered = _render(Qwen3ChatTemplate.template_str, messages)
    assert rendered == _render(qwen3_it_template, messages)
    assert "You are Qwen, created by Alibaba Cloud" not in rendered
