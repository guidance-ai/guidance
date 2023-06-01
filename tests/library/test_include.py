import guidance
import pytest

from ..utils import get_llm

SKIP_BASELINE_TESTS=True

@pytest.mark.skipif(SKIP_BASELINE_TESTS, reason="Does not test include tag; provides a baseline for comparison in the event of a regression.")
def test_guidance_capture_baseline():
    program = guidance(
        "It is {{context.holiday}}! {{input}} {{gen 'response'}}",
        llm=get_llm('transformers:gpt2')
    )
    output = program(
        input="What are some favorite pirate songs?",
        context=dict(holiday="Talk Like a Pirate Day"),
    )
    assert len(output["response"]) > 1, "Expected to capture response"


def test_guidance_capture_include():
    include_program = guidance("It is {{context.holiday}}!")
    program = guidance(
        "{{>include_program}} {{input}} {{gen 'response'}}",
        llm=get_llm('transformers:gpt2')
    )
    output = program(
        include_program=include_program,
        input="What are some favorite pirate songs?",
        context=dict(holiday="Talk Like a Pirate Day"),
    )
    assert len(output["response"]) > 1

@pytest.mark.skipif(SKIP_BASELINE_TESTS, reason="Does not test include tag; provides a baseline for comparison in the event of a regression.")
def test_guidance_capture_baseline():
    program = guidance(
        "{{#if context}}It is {{context.holiday}}! {{/if}}{{input}} {{gen 'response'}}",
        llm=get_llm('transformers:gpt2')
    )
    output = program(
        input="What are some favorite pirate songs?",
        context=dict(holiday="Talk Like a Pirate Day"),
    )
    assert len(output["response"]) > 1, "Expected to capture response"

def test_guidance_capture_include_with_if():
    include_program = guidance("{{#if context}}It is {{context.holiday}}! {{/if}}")
    program = guidance(
        "{{>include_program}}{{input}} {{gen 'response'}}",
        llm=get_llm('transformers:gpt2')
    )
    output = program(
        include_program=include_program,
        context=dict(holiday="Talk Like a Pirate Day"),
        input="What are some favorite pirate songs?",
    )
    assert len(output["response"]) > 1

def test_guidance_capture_include_output_with_if():
    include_program = guidance("{{#if context}}It is {{context.holiday}}! {{/if}}")
    include_output = include_program(context=dict(holiday="Talk Like a Pirate Day"))
    program = guidance(
        "{{>include_output}}{{input}} {{gen 'response'}}", 
        llm=get_llm('transformers:gpt2')
    )
    output = program(
        include_output=include_output,
        input="What are some favorite pirate songs?",
    ) 
    assert len(output["response"]) > 1, "Expected to capture response"
