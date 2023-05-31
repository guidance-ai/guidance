from typing import Any
from ._transformers import Transformers

class LlamaCpp2(Transformers):
    def __init__(self, model,
                 tokenizer = None,
                 caching=True, token_healing=True, acceleration=True,
                 temperature=0.0,
                 before_role: str = "<",
                 after_role: str = ">",
                 role_end: str = "",
                 include_role_in_end: bool = False,
                 before_role_end: str = "</",
                 after_role_end: str = ">", **kwargs):
        """ Llama.cpp model with guidance support.
        """
        
        assert acceleration == True, "LlamaCpp does not support acceleration=False"

        # TODO: @slundberg I think this should handled through subclasses like in the other models
        # there are too many corner cases that don't fit this pattern (for example uppercase letters or differences between roles)
        self.before_role = before_role
        self.after_role = after_role
        self.role_end = role_end
        self.include_role_in_end = include_role_in_end
        self.before_role_end = before_role_end
        self.after_role_end = after_role_end

        # make sure the right version of llama-cpp-python is installed
        try:
            from llama_cpp import Llama
            import pkg_resources
            from packaging import version
            assert version.parse(pkg_resources.get_distribution("llama-cpp-python").version) >= version.parse("0.1.55"), "llama-cpp-python version must be >= 0.1.55"
        except ImportError:
            raise ImportError("llama_cpp >= 0.1.55 must be installed to use the LlamaCpp model_obj in Gudiance! Install with `pip install llama-cpp-python`")
        
        # take strings and turn them into llama_cpp models
        if isinstance(model, str):
            model = Llama(
                model,
                **kwargs
            )
        
        # Create a tokenizer from the model if one wasn't passed
        if tokenizer is None:
            tokenizer = LlamaCppTokenizer(model)

        # wrap the llama_cpp model to act like a transformers model
        wrapped_model = LlamaCppInnerModel(model)

        # we are now ready to call the standard transformers init
        super().__init__(
            model=wrapped_model,
            tokenizer=tokenizer,
            caching=caching,
            token_healing=token_healing,
            acceleration=False, # llama_cpp supports acceleration by itself (TODO: to be confirmed)
            temperature=temperature
        )

class LlamaCppTokenizer():
    """This simulates a HuggingFace tokenizer for llama_cpp models."""

    def __init__(self, model_obj):
        from llama_cpp import llama_n_vocab, llama_token_eos
        self.model_obj = model_obj
        self.vocab_size = llama_n_vocab(self.model_obj.ctx)
        self.eos_token_id = llama_token_eos()
        self.eos_token = self.decode([self.eos_token_id])

    def __call__(self, string, **kwargs):
        return self.encode(string, **kwargs)

    def encode(self, string, **kwargs):
        return self.model_obj.tokenize(string)
    
    def decode(self, tokens, **kwargs):
        return self.model_obj.detokenize(tokens)
    
    def convert_ids_to_tokens(self, ids):
        return [self.decode([id]) for id in ids]
    
    def convert_tokens_to_ids(self, tokens):
        return [self.encode(token)[0] for token in tokens]
    
class LlamaCppInnerModel():
    """This simulates a HuggingFace model for llama_cpp models."""

    def __init__(self, llama_model):
        self.llama_model = llama_model

        from llama_cpp import llama_n_vocab
        self.config = LlamaCppInnerModelConfig()
        self.config.vocab_size = llama_n_vocab(self.llama_model.ctx)
        self.config.max_sequence_length = self.llama_model.params.n_ctx
        self.config.pad_token_id = None
        self.device = None # this stops the transformers code from trying to move the model to a device
    
    def generate(self, inputs, attention_mask, temperature, max_new_tokens, top_p, pad_token_id, logits_processor, stopping_criteria,
                 output_scores, return_dict_in_generate, do_sample, streamer):
        
        # TODO: this is not all working yet :)
        return self.llama_model.create_completion(
            inputs=inputs,
            attention_mask=attention_mask,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            pad_token_id=pad_token_id,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            do_sample=do_sample,
            streamer=streamer
        )

class LlamaCppInnerModelConfig():
    pass