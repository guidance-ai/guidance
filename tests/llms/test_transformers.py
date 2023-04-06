
def test_basic():

    import guidance

    llm = guidance.llms.Transformers('gpt2')
    with llm.session() as s:
        out = s("this is a test", max_tokens=5)
        print(out)
    