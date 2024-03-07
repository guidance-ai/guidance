import guidance
import pytest

def test_peft():
    try:
        import peft
    except:
        raise Exception("Sorry, peft is not installed")
