def test_experimental():
    from huggingface_hub import hf_hub_download
    from guidance import models, system, user, assistant, gen

    from guidance.visual import JupyterWidgetRenderer
    from guidance._singleton import set_renderer, get_trace_handler

    trace_handler = get_trace_handler()
    set_renderer(JupyterWidgetRenderer(trace_handler))

    model_path = hf_hub_download("lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF",
                                 "Phi-3.1-mini-4k-instruct-IQ3_M.gguf")
    model = models.LlamaCpp(model_path)

    lm = model
    with system():
        lm += "Talk like a pirate!"
    with user():
        lm += "Hello, model!"
    with assistant():
        lm += gen()
    with user():
        lm += "What is the capital of France?"
    with assistant():
        lm += gen()