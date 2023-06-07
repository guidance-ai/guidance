import guidance

def test_clear():
    """Makes sure we call clear"""
    guidance.llms.OpenAI.cache.clear()