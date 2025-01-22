def test_repeat_base_qwen_widget():
    from guidance.models import Transformers
    from guidance import gen
    from guidance.visual import JupyterWidgetRenderer

    model = "Qwen/Qwen2-1.5B-Instruct"
    base_lm = Transformers(model)
    old_renderer = base_lm.engine.renderer
    base_lm.engine.renderer = JupyterWidgetRenderer(base_lm.engine.trace_handler)
    del old_renderer

    # This always passes
    lm = base_lm + """\
    1 + 1 = add(1, 1) = 2
    3 + 5 = add(3, 5) = 8
    11 + 9 = """
    lm = lm + gen(max_tokens=33, name="result")

    # We're testing this second attempt runs or raises exception.
    from guidance import guidance
    @guidance
    def add(lm, input1, input2):
        lm += f" = {int(input1) + int(input2)}"
        return lm

    lm = base_lm + """\
    1 + 1 = add(1, 1) = 2
    3 + 5 = add(3, 5) = 8
    11 + 9"""
    lm = lm + gen(max_tokens=30, tools=[add])

    assert True