import os
import re
import textwrap

from typing import Sequence, Union

try:
    import torch
except ModuleNotFoundError:
    pass


try:
    import transformers as transformers_package

    has_transformers = True
except ModuleNotFoundError:
    has_transformers = False


from .._model import Engine, Model
from .._tokenizer import Tokenizer


class TransformersTokenizer(Tokenizer):
    def __init__(
        self,
        model,
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast",
            None,
        ],
        chat_template=None,
        ignore_bos_token=False,
    ):
        if transformers_tokenizer is None:
            transformers_tokenizer = self._tokenizer(model)
        else:
            is_ptt = isinstance(transformers_tokenizer, transformers_package.PreTrainedTokenizer)
            is_ptt_fast = isinstance(
                transformers_tokenizer, transformers_package.PreTrainedTokenizerFast
            )
            assert is_ptt or is_ptt_fast

        self._orig_tokenizer = transformers_tokenizer
        special_tokens_map = {
            id: token for token, id in transformers_tokenizer.get_added_vocab().items()
        }

        # build out the set of byte_string tokens
        byte_tokens = [b""] * len(transformers_tokenizer)
        if hasattr(transformers_tokenizer, "byte_decoder"):
            byte_decoder = transformers_tokenizer.byte_decoder

            for i in range(len(transformers_tokenizer)):
                byte_coded = bytes(
                    [byte_decoder[c] for c in transformers_tokenizer.convert_ids_to_tokens(i)]
                )
                byte_tokens[i] = byte_coded

        elif hasattr(transformers_tokenizer, "sp_model"):
            space_prefix = "▁".encode()
            for i in range(len(transformers_tokenizer)):
                if i in special_tokens_map:
                    byte_coded = special_tokens_map[i].encode()
                else:
                    byte_coded = re.sub(
                        rb"<0x(..)>",
                        lambda x: bytes.fromhex(x[1].decode()),
                        transformers_tokenizer.sp_model.id_to_piece(i).encode(),
                    )
                byte_tokens[i] = byte_coded.replace(space_prefix, b" ")

        else:
            byte_decoder = transformers_package.AutoTokenizer.from_pretrained(
                "gpt2", use_fast=False
            ).byte_decoder  # fall back to gpt2 mapping

            # some special tokens may not have their whitespace encoded...
            byte_decoder[" "] = 32
            byte_decoder["\n"] = 10
            byte_decoder["\r"] = 13
            byte_decoder["\t"] = 9
            byte_decoder["▁"] = 32

            # run a quick spot check to verify we can rebuild complex multi-token unicode symbols
            s = "’•¶∂ƒ˙∆£Ħ爨ൠᅘ∰፨"
            reconstructed = b""
            try:

                for i in transformers_tokenizer(s)["input_ids"]:
                    nxt_bytes = []
                    for c in transformers_tokenizer.convert_ids_to_tokens(i):
                        nxt_bytes.append(byte_decoder[c])
                    reconstructed += bytes(nxt_bytes)
                # Check if the tokenizer has a bos_token attribute, and if it does, check
                # if it's at the start of the reconstructed bytes
                # Some tokenizers add this automatically as part of the call function, so
                # we need to remove it to compare
                if hasattr(transformers_tokenizer, "bos_token") and reconstructed.startswith(
                    transformers_tokenizer.bos_token.encode()
                ):
                    reconstructed = reconstructed[len(transformers_tokenizer.bos_token) :]
            except Exception as e:
                msg = textwrap.dedent(
                    f"""
                    The tokenizer being used is unable to convert a special character in {s}.
                    For models with sentencepiece based tokenizers (e.g. llama, phi-3-mini),
                    installing sentencepiece often fixes this issue (pip install sentencepiece).
                    """
                )
                raise ValueError(msg) from e
            assert (
                reconstructed.decode() == s
            ), "The passed tokenizer does not have a byte_decoder property and using a standard gpt2 byte_decoder fails!"

            for i in range(len(transformers_tokenizer)):
                byte_coded = bytes(
                    [byte_decoder[c] for c in transformers_tokenizer.convert_ids_to_tokens(i)]
                )
                byte_tokens[i] = byte_coded

        # Chat Template logic
        if chat_template is None and hasattr(self._orig_tokenizer, "chat_template"):
            chat_template = self._orig_tokenizer.chat_template

        # the superclass does most of the work once we have the tokens
        super().__init__(
            byte_tokens,
            chat_template,
            None if ignore_bos_token else transformers_tokenizer.bos_token_id,
            transformers_tokenizer.eos_token_id,
        )

    def _tokenizer(self, model, **kwargs) -> Union[
        "transformers_package.PreTrainedTokenizer",
        "transformers_package.PreTrainedTokenizerFast",
    ]:
        # intantiate the tokenizer
        if isinstance(model, str):
            # make sure transformers is installed
            if not has_transformers:
                raise Exception("Please install transformers with `pip install transformers`")

            try:
                tokenizer = transformers_package.AutoTokenizer.from_pretrained(
                    model, use_fast=False, **kwargs
                )
                # This is here because some tokenizers are bad and don't have all the bytes (I'm looking at you, microsoft/phi2)
                if hasattr(tokenizer, "byte_decoder"):
                    all_bytes = set()
                    for x in tokenizer.get_vocab().keys():
                        for y in x:
                            all_bytes.add(y)
                    assert set(tokenizer.byte_decoder.keys()).intersection(all_bytes) == all_bytes
            except:
                tokenizer = transformers_package.AutoTokenizer.from_pretrained(
                    model, use_fast=True, **kwargs
                )  # fall back to the fast tokenizer

        assert (
            tokenizer is not None
        ), "You must give a model name when you provide a tokenizer object!"

        return tokenizer

    def encode(self, byte_string: bytes) -> Sequence[int]:
        assert isinstance(byte_string, bytes)
        # HF tokenizers take in strings apparently
        tokenization = self._orig_tokenizer(byte_string.decode(), add_special_tokens=False)
        return tokenization["input_ids"]

    def decode(self, tokens: Sequence[int]) -> bytes:
        decoded_str = self._orig_tokenizer.decode(tokens)
        return decoded_str.encode()

    def recode(self, tokens: Sequence[int]) -> Sequence[int]:
        # the encode/decode cycle might not work if we have partial unicode strings
        used_tokens = len(tokens)
        for _ in range(3):
            try:
                first_decode = self.decode(tokens).decode("utf8")
            except UnicodeDecodeError:
                if used_tokens == 0:
                    break
                else:
                    used_tokens -= 1

        new_ids = list(self.encode(first_decode.encode("utf-8")))
        if used_tokens < len(tokens):
            new_ids += tokens[used_tokens:]

        # HACK: check for a bug in the HuggingFace tokenizer
        # (that will just add extra spaces during an encode-decode cycle)
        second_decode = self._orig_tokenizer.decode(new_ids)
        if (
            second_decode != first_decode
            and len(second_decode) == len(first_decode) + 1
            and second_decode.startswith("<s>  ")
        ):
            new_ids = new_ids[0:1] + new_ids[2:]

        return new_ids


class TransformersEngine(Engine):
    def __init__(self, model, tokenizer, compute_log_probs: bool, chat_template=None, **kwargs):
        # fill in default model value
        if model is None:
            model = os.environ.get("TRANSFORMERS_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser("~/.transformers_model"), "r") as file:
                    model = file.read().replace("\n", "")
            except:
                pass

        self.model_obj = self._model(model, **kwargs)

        if not isinstance(model, str):
            self.model = model.__class__.__name__
        self.device = self.model_obj.device  # otherwise note the current device

        self._past_key_values = None
        self._cached_logits = None
        self._cached_token_ids: list[int] = []

        # Set attr for malformed tokenizer hack.
        # If more models start doing this, generalize into a util function.
        if hasattr(self.model_obj.config, "model_type"):
            if self.model_obj.config.model_type in ["phi3"]:
                self._disable_retokenize_check = True

        super().__init__(
            TransformersTokenizer(model, tokenizer, chat_template),
            compute_log_probs=compute_log_probs,
        )
        assert self._token_trie.match

    def _model(self, model, **kwargs):
        # intantiate the model if needed
        if isinstance(model, str):

            # make sure transformers is installed
            if not has_transformers:
                raise Exception(
                    "Please install transformers with `pip install transformers` in order to use guidance.models.Transformers!"
                )
            model = transformers_package.AutoModelForCausalLM.from_pretrained(model, **kwargs)
        return model

    def get_logits(self, token_ids, forced_bytes, current_temp):
        """Computes the logits for the given token state.

        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        """

        # make sure we don't run off the end of the model
        if len(token_ids) >= getattr(self.model_obj.config, "max_position_embeddings", 1e10):
            raise Exception(
                f"Attempted to run a transformers model past its maximum context window size of {self.model_obj.config.max_position_embeddings}!"
            )

        # get the number of cache positions we are using
        cache_token_ids = self._cached_token_ids
        num_cached = 0
        for id in cache_token_ids:
            if (
                num_cached >= len(cache_token_ids)
                or num_cached >= len(token_ids)
                or token_ids[num_cached] != id
            ):
                break
            num_cached += 1

        # reset the cache length according to that number of positions
        past_key_values = self._past_key_values
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        if past_length > num_cached:
            # note we recompute the last token because we don't bother to handle the special case of just computing logits
            past_length = max(0, num_cached - 1)
            self._past_key_values = tuple(
                tuple(p[..., :past_length, :] for p in v) for v in past_key_values
            )
        cache_token_ids[past_length:] = []

        # call the model
        new_token_ids = token_ids[past_length:]
        if len(new_token_ids) > 0:
            with torch.no_grad():
                model_out = self.model_obj(
                    input_ids=torch.tensor(new_token_ids).unsqueeze(0).to(self.device),
                    past_key_values=self._past_key_values,
                    use_cache=True,
                    position_ids=torch.arange(past_length, past_length + len(new_token_ids))
                    .unsqueeze(0)
                    .to(self.device),
                    attention_mask=torch.ones(1, past_length + len(new_token_ids)).to(self.device),
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

            # save the results
            self._past_key_values = model_out.past_key_values
            cache_token_ids.extend(new_token_ids)
            # Need to add special truncating logic here for weird models that have a different output size than tokenizer vocab
            self._cached_logits = (
                model_out.logits[0, -1, : len(self.tokenizer.tokens)].cpu().numpy()
            )
            self.metrics.engine_input_tokens += len(new_token_ids)
            self.metrics.engine_output_tokens += 1

        return self._cached_logits


class Transformers(Model):
    def __init__(
        self,
        model=None,
        tokenizer=None,
        echo=True,
        compute_log_probs=False,
        chat_template=None,
        **kwargs,
    ):
        """Build a new Transformers model object that represents a model in a given state."""
        super().__init__(
            TransformersEngine(
                model,
                tokenizer,
                compute_log_probs,
                chat_template=chat_template,
                **kwargs,
            ),
            echo=echo,
        )
