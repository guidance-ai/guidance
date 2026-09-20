import inspect
import warnings

from guidance.chat import Phi4MiniChatTemplate, load_template_class
from guidance.models._engine import Tokenizer
from guidance.models._engine._tokenizer import TokenizerWrappable


def _as_instance(chat_template):
    if inspect.isclass(chat_template):
        return chat_template()
    return chat_template


def test_load_template_class_from_instance():
    template = Phi4MiniChatTemplate()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = load_template_class(template)
    chat_template = _as_instance(loaded)
    assert chat_template.get_role_start("user") == template.get_role_start("user")
    assert chat_template.get_role_end() == template.get_role_end()


def test_tokenizer_chat_template_from_instance():
    tokens = [bytes([i]) for i in range(256)]
    ll_tokenizer = TokenizerWrappable(
        eos_token_id=0,
        bos_token_id=0,
        tokens=tokens,
        special_token_ids=[0],
        encode_callable=lambda byte_string: list(byte_string),
    ).as_ll_tokenizer()
    tokenizer = Tokenizer(ll_tokenizer=ll_tokenizer, chat_template=Phi4MiniChatTemplate())
    chat_template = _as_instance(tokenizer.chat_template)
    assert chat_template.get_role_start("user") == "<|user|>"
    assert chat_template.get_role_end() == "<|end|>"
