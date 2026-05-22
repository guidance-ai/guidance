"""Tests for guidance.chat module — focusing on load_template_class() name lookup."""

import warnings

import pytest

from guidance.chat import (
    CHAT_TEMPLATE_NAME_MAP,
    ChatMLTemplate,
    ChatTemplate,
    Gemma29BInstructChatTemplate,
    Llama2ChatTemplate,
    Llama3ChatTemplate,
    Llama3dot2ChatTemplate,
    Mistral7BInstructChatTemplate,
    Phi3MiniChatTemplate,
    Phi3SmallMediumChatTemplate,
    Phi4MiniChatTemplate,
    Qwen2dot5ChatTemplate,
    Qwen3ChatTemplate,
    load_template_class,
)


class TestLoadTemplateClassByName:
    """load_template_class() should resolve friendly names case-insensitively."""

    @pytest.mark.parametrize(
        "name,expected_class",
        [
            # ChatML variants
            ("chatml", ChatMLTemplate),
            ("ChatML", ChatMLTemplate),
            ("CHATML", ChatMLTemplate),
            ("chat-ml", ChatMLTemplate),
            ("chat_ml", ChatMLTemplate),
            # Llama 2
            ("llama2", Llama2ChatTemplate),
            ("Llama2", Llama2ChatTemplate),
            ("llama-2", Llama2ChatTemplate),
            ("llama_2", Llama2ChatTemplate),
            # Llama 3
            ("llama3", Llama3ChatTemplate),
            ("Llama3", Llama3ChatTemplate),
            # Phi-3
            ("phi3", Phi3MiniChatTemplate),
            ("phi3mini", Phi3MiniChatTemplate),
            ("Phi3Mini", Phi3MiniChatTemplate),
            ("phi-3-mini", Phi3MiniChatTemplate),
            ("phi3small", Phi3SmallMediumChatTemplate),
            ("phi3medium", Phi3SmallMediumChatTemplate),
            # Phi-4
            ("phi4mini", Phi4MiniChatTemplate),
            ("phi4", Phi4MiniChatTemplate),
            # Mistral
            ("mistral", Mistral7BInstructChatTemplate),
            ("mistral7b", Mistral7BInstructChatTemplate),
            ("Mistral", Mistral7BInstructChatTemplate),
            # Gemma 2
            ("gemma2", Gemma29BInstructChatTemplate),
            ("gemma29b", Gemma29BInstructChatTemplate),
            ("Gemma2", Gemma29BInstructChatTemplate),
            # Qwen 2.5
            ("qwen25", Qwen2dot5ChatTemplate),
            ("qwen2dot5", Qwen2dot5ChatTemplate),
            ("Qwen25", Qwen2dot5ChatTemplate),
            # Qwen 3
            ("qwen3", Qwen3ChatTemplate),
            ("Qwen3", Qwen3ChatTemplate),
            # Llama 3.2
            ("llama32", Llama3dot2ChatTemplate),
            ("llama3dot2", Llama3dot2ChatTemplate),
            ("Llama32", Llama3dot2ChatTemplate),
            ("llama-3.2", Llama3dot2ChatTemplate),
            ("Llama 3.2", Llama3dot2ChatTemplate),
        ],
    )
    def test_friendly_name_lookup(self, name, expected_class):
        result = load_template_class(name)
        assert result is expected_class

    def test_none_returns_chatml(self):
        result = load_template_class(None)
        assert result is ChatMLTemplate

    def test_class_passed_directly(self):
        result = load_template_class(Llama2ChatTemplate)
        assert result is Llama2ChatTemplate

    def test_base_class_raises(self):
        with pytest.raises(Exception, match="base ChatTemplate class"):
            load_template_class(ChatTemplate)

    def test_unknown_name_warns_and_falls_back(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = load_template_class("totally_unknown_model_xyz")
        assert result is ChatMLTemplate
        assert len(caught) == 1
        assert "totally_unknown_model_xyz" in str(caught[0].message)

    def test_name_map_keys_are_normalized(self):
        """All keys in CHAT_TEMPLATE_NAME_MAP must already be normalized (lowercase, no separators/dots)."""
        for key in CHAT_TEMPLATE_NAME_MAP:
            normalized = key.lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
            assert key == normalized, f"Key {key!r} is not normalized — should be {normalized!r}"

    def test_name_map_values_are_chat_template_subclasses(self):
        """All values in CHAT_TEMPLATE_NAME_MAP must be ChatTemplate subclasses."""
        for key, cls in CHAT_TEMPLATE_NAME_MAP.items():
            assert isinstance(cls, type) and issubclass(cls, ChatTemplate), (
                f"Value for key {key!r} is not a ChatTemplate subclass"
            )
