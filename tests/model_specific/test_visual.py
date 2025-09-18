from guidance.registry import get_renderer


def test_repeat_simple_model():
    from guidance import gen
    from guidance.models import Transformers
    from guidance.registry import get_trace_handler, set_renderer
    from guidance.visual import JupyterWidgetRenderer

    trace_handler = get_trace_handler()
    original_renderer = get_renderer()
    for _ in range(2):
        set_renderer(JupyterWidgetRenderer(trace_handler))

        lm = Transformers("gpt2")
        lm += "Hi hi hi"
        lm += gen(max_tokens=5)

        set_renderer(original_renderer)

    assert True


def test_roles():
    from guidance import gen, system, user
    from guidance.models import Transformers

    m0 = Transformers("gpt2")
    with system():
        m1 = m0 + "You are responsible for writing an epic poem."
    with user():
        m2 = m1 + "Roses are red and " + gen(name="suffix", regex=r"[\w\s]{20,30}", max_tokens=30)

    assert m2 is not None


def test_divergence():
    from guidance import gen
    from guidance.models import Transformers
    from guidance.registry import get_trace_handler, set_renderer
    from guidance.visual import JupyterWidgetRenderer

    trace_handler = get_trace_handler()
    original_renderer = get_renderer()
    try:
        m0 = Transformers("gpt2")
        set_renderer(JupyterWidgetRenderer(trace_handler))
        m1 = m0 + "Give me some out:\n"
        m2 = m1 + gen(max_tokens=5)
        m3 = m1 + gen(max_tokens=5)
    except Exception as e:
        raise e
    finally:
        set_renderer(original_renderer)

    assert True
