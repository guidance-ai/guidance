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


def test_repeat_simple_model():
    from guidance.models import Transformers
    from guidance import gen
    from guidance.registry.__init__ import set_renderer, get_trace_handler
    from guidance.visual import JupyterWidgetRenderer

    for i in range(2):
        trace_handler = get_trace_handler()
        set_renderer(JupyterWidgetRenderer(trace_handler))

        lm = Transformers('gpt2')
        lm += 'Hi hi hi'
        lm += gen(max_tokens=5)

    assert True


def test_roles():
    from guidance.models import Transformers
    from guidance import gen, user, system

    m0 = Transformers("gpt2")
    with system():
        m1 = m0 + "You are responsible for writing an epic poem."
    with user():
        m2 = m1 + "Roses are red and " + gen(name="suffix", regex=r'[\w\s]{20,30}', max_tokens=30)

    assert m2 is not None