import os
import time
import collections
import regex
import pygtrie
import queue
import threading
import logging
from .._llm import LLM, LLMSession, SyncSession
from .._transformers import Transformers, TransformersSession

class MPT(Transformers):
    """ A HuggingFace transformers version of the MosaicML MPT language model with Guidance support.
    """

    cache = LLM._open_cache("_mpt.diskcache")

    def _model_and_tokenizer(self, model, tokenizer, **kwargs):

        # load the MPT specific tokenizer and model
        import transformers
        if isinstance(model, str):

            # MPT uses the same tokenizer as GPT-NeoX
            if tokenizer is None:
                tokenizer = tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", **kwargs)
            
            dynamic_kwargs = {}

            # use triton for attention
            config = transformers.AutoConfig.from_pretrained(
                model,
                trust_remote_code=True
            )

            # allow for a custom attention implementation
            if kwargs.get("attn_impl", None) is not None:
                import torch
                config.attn_config['attn_impl'] = kwargs["attn_impl"]
                dynamic_kwargs["torch_dtype"] = torch.bfloat16

            # allow for a custom max_seq_len (enabled by ALiBi)
            if kwargs.get("max_seq_len", None) is not None:
                config.update({"max_seq_len": kwargs["max_seq_len"]})

            model = transformers.AutoModelForCausalLM.from_pretrained(
                model,
                config=config,
                trust_remote_code=True,
                **kwargs
            )
            
        return super()._model_and_tokenizer(model, tokenizer, **kwargs)

class MPTChat(MPT):

    default_system_prompt = """- You are a helpful assistant chatbot trained by MosaicML.  
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes."""

    @staticmethod
    def role_start(role):
       return  {
        'user': '<|im_start|>user ',
        'system': '<|im_start|>system\n',
        'assistant': '<|im_start|>assistant ',
        }[role]
    
    @staticmethod
    def role_end(role):
        return '<|im_end|>'