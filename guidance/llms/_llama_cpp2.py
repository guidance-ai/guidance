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
        self.eos_token = "</s>" # TODO: is there a way to get this from llama-cpp-python?

    def encode(self, string, **kwargs):
        return self.model_obj.tokenize(string.encode("utf-8"))
    
    def decode(self, tokens, **kwargs):
        return self.model_obj.detokenize(tokens).decode("utf-8", errors="ignore") # errors="ignore" is copied from llama-cpp-python
    
    def convert_ids_to_tokens(self, ids):
        return [self.decode([id]) for id in ids]
    
    def convert_tokens_to_ids(self, tokens):
        return [self.encode(token)[0] for token in tokens]
    
    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)
    
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
    
    def generate(self, inputs, temperature, max_new_tokens, top_p, pad_token_id, logits_processor, stopping_criteria,
                 output_scores, return_dict_in_generate, streamer, do_sample=None):
        
        assert len(inputs) == 1, "LlamaCpp only supports one input sequence at a time, so n > 1 is not supported right now."

        import torch
        
        # gen_args = {}
        # if do_sample is not None:
        #     gen_args["do_sample"] = do_sample
        
        # TODO: this is not all working yet :)
        token_generator = self.llama_model.generate(
            tokens=inputs[0],
            temp=temperature,
            # max_new_tokens=max_new_tokens,
            top_p=top_p,
            # pad_token_id=pad_token_id,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            # output_scores=output_scores,
            # return_dict_in_generate=return_dict_in_generate,
            # do_sample=do_sample,
            # streamer=streamer,
            # **gen_args
        )

        # to match the ways transformers works we add the input sequence to the beginning of the output
        streamer.put({
            "sequences": inputs
        })

        tokens = []
        scores = []
        i = 0
        for token in token_generator:
            tokens.append(token)
            scores.append(self.llama_model.eval_logits[-1])
            if streamer is not None:
                streamer.put({
                    "sequences": torch.tensor(tokens).unsqueeze(0),
                    "scores": scores,
                })
                tokens = []
                scores = []

            if i >= max_new_tokens:
                break
        
        if len(tokens) > 0:
            return {
                "sequences": torch.tensor(tokens).unsqueeze(0),
                "scores": scores
            }

class LlamaCppInnerModelConfig():
    pass