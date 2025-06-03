import os
import re
import textwrap
import warnings
from typing import TYPE_CHECKING, Optional, Union, cast

from ..chat import ChatTemplate
from ._engine import Engine, Tokenizer, EngineInterpreter, Llama3VisionInterpreter, Phi3VisionInterpreter
from ._engine._tokenizer import TokenizerWrappable
from ._base import Model

try:
    import torch
except ModuleNotFoundError:
    pass

try:
    import transformers as transformers_package

    has_transformers = True
except ModuleNotFoundError:
    has_transformers = False
else:
    import llguidance.hf

if TYPE_CHECKING:
     from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

# Formed by comparing model and tokenizer from_pretrained methods
# transformers/models/auto/auto_factory.py
# transformers/models/auto/tokenization_auto.py
_COMMON_TRANSFORMERS_KWARGS = [
    "cache_dir",
    "force_download",
    "proxies",
    "resume_download",
    "revision",
    "subfolder",
    "trust_remote_code",
]


class ByteDecoderError(Exception):
    pass


class ByteTokensError(Exception):
    pass


class TransformersTokenizer(Tokenizer):
    def __init__(
        self,
        hf_tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
        chat_template: Union[str, ChatTemplate, None] = None,
    ) -> "TransformersTokenizer":
        self._orig_tokenizer = hf_tokenizer

        if isinstance(hf_tokenizer, transformers_package.PreTrainedTokenizerFast):
            ll_tokenizer = llguidance.hf.from_tokenizer(
                hf_tokenizer=hf_tokenizer,
            )
            vocab_size = hf_tokenizer.backend_tokenizer.get_vocab_size(with_added_tokens=True)
        else:
            byte_tokens = self._byte_tokens(hf_tokenizer)
            vocab_size = len(byte_tokens)
            ll_tokenizer = TokenizerWrappable(
                eos_token_id=hf_tokenizer.eos_token_id, # type: ignore[attr-defined]
                bos_token_id=getattr(hf_tokenizer, "bos_token_id", None),
                tokens=byte_tokens,
                special_token_ids=[
                    token_id for (token_id, token) in hf_tokenizer.added_tokens_decoder.items()
                    if token.special
                ],
                encode_callable=hf_tokenizer.encode,
            ).as_ll_tokenizer()
        super().__init__(
            ll_tokenizer=ll_tokenizer,
            chat_template=chat_template,
            bos_token_id=getattr(hf_tokenizer, "bos_token_id", None),
        )
        self._vocab_size = vocab_size

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        chat_template: Union[str, ChatTemplate, None] = None,
        use_fast=True,
        **kwargs
    ) -> "TransformersTokenizer":
        tokenizer = transformers_package.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=use_fast,
            **kwargs
        )
        return cls(
            hf_tokenizer=tokenizer,
            chat_template=chat_template,
        )


    def recode(self, tokens: list[int]) -> list[int]:
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

    @classmethod
    def _byte_tokens(
        cls,
        transformers_tokenizer: Union[
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
        ],
    ) -> list[bytes]:

        if hasattr(transformers_tokenizer, "byte_decoder"):
            try:
                cls._check_byte_decoder(
                    transformers_tokenizer.byte_decoder, transformers_tokenizer
                )
            except ByteDecoderError as e:
                warnings.warn(
                    f"Tokenizer has a byte_decoder, but it can't be used to construct byte_tokens: {e}"
                )
                pass
            else:
                return cls._byte_tokens_from_byte_decoder(
                    transformers_tokenizer.byte_decoder, transformers_tokenizer
                )

        if hasattr(transformers_tokenizer, "sp_model"):
            return cls._byte_tokens_from_sp_model(transformers_tokenizer)

        try:
            return cls._byte_tokens_by_encoding_token_strings(transformers_tokenizer)
        except ValueError as e:
            warnings.warn(
                f"Could not build_byte tokens from the tokenizer by encoding token strings: {e}"
            )
            pass

        fallback_byte_decoder = cls._fallback_byte_decoder()
        try:
            cls._check_byte_decoder(fallback_byte_decoder, transformers_tokenizer)
        except ByteDecoderError as e:
            # Should be the only exception that is raised in _byte_tokens
            raise ByteTokensError(
                "Could not build byte tokens from the tokenizer, and falling back to a standard gpt2 byte_decoder failed"
            ) from e
        return cls._byte_tokens_from_byte_decoder(fallback_byte_decoder, transformers_tokenizer)

    @classmethod
    def _byte_tokens_from_byte_decoder(
        cls,
        byte_decoder: dict[str, int],
        transformers_tokenizer: Union[
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
        ],
    ) -> list[bytes]:
        byte_tokens = [b""] * len(transformers_tokenizer)
        for i in range(len(transformers_tokenizer)):
            byte_coded = bytes(
                [byte_decoder[c] for c in transformers_tokenizer.convert_ids_to_tokens(i)]
            )
            byte_tokens[i] = byte_coded
        return byte_tokens

    @classmethod
    def _byte_tokens_from_sp_model(
        cls,
        transformers_tokenizer: Union[
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
        ],
    ) -> list[bytes]:
        byte_tokens = [b""] * len(transformers_tokenizer)
        special_tokens_map = {
            id: token for token, id in transformers_tokenizer.get_added_vocab().items()
        }
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
        return byte_tokens

    @classmethod
    def _byte_tokens_by_encoding_token_strings(
        cls,
        transformers_tokenizer: Union[
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
        ],
    ) -> list[bytes]:
        byte_tokens = [b""] * len(transformers_tokenizer)
        special_tokens_map = {
            id: token for token, id in transformers_tokenizer.get_added_vocab().items()
        }
        byte_encoder = cls._bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        for i in range(len(transformers_tokenizer)):
            if i in special_tokens_map:
                byte_coded = special_tokens_map[i].encode()
            else:
                token = transformers_tokenizer.convert_ids_to_tokens(i)
                if isinstance(token, bytes):
                    byte_coded = token
                elif isinstance(token, str):
                    if hasattr(transformers_tokenizer, "convert_tokens_to_string"):
                        token_str = transformers_tokenizer.convert_tokens_to_string([token])
                        encoded_str = transformers_tokenizer.encode(token_str)
                        if len(encoded_str) != 1:
                            raise ValueError(
                                f"Round-trip encoding of tokens [{token}] failed! Got {encoded_str}"
                            )
                        roundtrip_id = encoded_str[0]
                        if roundtrip_id == i:
                            byte_coded = token_str.encode()
                        else:
                            byte_coded = bytes([byte_decoder[c] for c in token])
                    else:
                        byte_coded = token.encode()
                else:
                    raise ValueError(f"Unexpected token type: {type(token)}")
            byte_tokens[i] = byte_coded
        return byte_tokens

    @classmethod
    def _fallback_byte_decoder(cls) -> dict[str, int]:
        byte_decoder = transformers_package.AutoTokenizer.from_pretrained(
            "gpt2", use_fast=False
        ).byte_decoder  # fall back to gpt2 mapping

        # some special tokens may not have their whitespace encoded...
        byte_decoder[" "] = 32
        byte_decoder["\n"] = 10
        byte_decoder["\r"] = 13
        byte_decoder["\t"] = 9
        byte_decoder["▁"] = 32

        return byte_decoder

    @classmethod
    def _check_byte_decoder(
        cls,
        byte_decoder: dict[str, int],
        transformers_tokenizer: Union[
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
        ],
    ) -> None:

        def check_byte_decoder_has_all_bytes() -> None:
            # This is here because some tokenizers are bad and don't have all the bytes (I'm looking at you, microsoft/phi2)
            all_bytes = set()
            for x in transformers_tokenizer.get_vocab().keys():
                for y in x:
                    all_bytes.add(y)
            if not set(byte_decoder.keys()) >= all_bytes:
                raise ByteDecoderError(
                    f"Byte decoder is missing bytes: {all_bytes - set(byte_decoder.keys())}"
                )

        def check_byte_decoder_complex_round_trip() -> None:
            # run a quick spot check to verify we can rebuild complex multi-token unicode symbols
            s = "’•¶∂ƒ˙∆£Ħ爨ൠᅘ∰፨"
            reconstructed = b""
            try:
                input_ids = transformers_tokenizer(s)["input_ids"]
                for i in input_ids:
                    nxt_bytes = []
                    token_str = transformers_tokenizer.convert_ids_to_tokens(i)
                    for c in token_str:
                        nxt_bytes.append(byte_decoder[c])
                    reconstructed += bytes(nxt_bytes)
                # Check if the tokenizer has a bos_token attribute, and if it does, check
                # if it's at the start of the reconstructed bytes
                # Some tokenizers add this automatically as part of the call function, so
                # we need to remove it to compare
                if (
                    hasattr(transformers_tokenizer, "bos_token")
                    and transformers_tokenizer.bos_token
                    and reconstructed.startswith(transformers_tokenizer.bos_token.encode())
                ):
                    reconstructed = reconstructed[len(transformers_tokenizer.bos_token) :]
            # TODO: can we narrow this exception?
            except Exception as e:
                msg = textwrap.dedent(
                    f"""
                    The tokenizer being used is unable to convert a special character in {s}.
                    For models with sentencepiece based tokenizers (e.g. llama, phi-3-mini),
                    installing sentencepiece often fixes this issue (pip install sentencepiece).
                    """
                )
                raise ByteDecoderError(msg) from e
            if reconstructed.decode() != s:
                raise ByteDecoderError(
                    f"Failed to reconstruct the string {s} from the tokenizer's byte_decoder: {reconstructed.decode()!r} != {s!r}"
                )

        check_byte_decoder_has_all_bytes()
        check_byte_decoder_complex_round_trip()

    @classmethod
    def _bytes_to_unicode(cls):
        bs = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

class TransformersEngine(Engine):
    def __init__(
        self,
        model: Union[str, "PreTrainedModel"],
        tokenizer: Union[
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
            None,
        ],
        compute_log_probs: bool,
        chat_template=None,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        **kwargs,
    ):
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
            try:
                self.model = self.model_obj.config._name_or_path
            except AttributeError:
                self.model = self.model_obj.__class__.__name__
        else:
            self.model = model
        self.device = self.model_obj.device  # otherwise note the current device

        self._past_key_values: Union[
            transformers_package.Cache,
            tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
            None,
        ] = None
        self._cached_logits = None
        self._cached_token_ids: list[int] = []

        # Set attr for malformed tokenizer hack.
        # If more models start doing this, generalize into a util function.
        if hasattr(self.model_obj.config, "model_type"):
            if self.model_obj.config.model_type in ["phi3"]:
                self._disable_retokenize_check = True

        # Automatically fill common args between Transformers
        # model and tokenizer
        passed_common_kwargs = {}
        for arg_name in _COMMON_TRANSFORMERS_KWARGS:
            if arg_name in kwargs:
                passed_common_kwargs[arg_name] = kwargs[arg_name]

        # Create the tokenizer
        if tokenizer is None:
            if isinstance(model, str):
                # Note: prioritize the fast tokenizer if available
                tokenizer = cast(
                    Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
                    transformers_package.AutoTokenizer.from_pretrained(
                        pretrained_model_name_or_path=model,
                        use_fast=True,
                        **kwargs
                    )
                )
            else:
                raise ValueError(
                    "If no tokenizer is provided, a model name must be provided to load the tokenizer."
                )
        my_tokenizer = TransformersTokenizer(
            hf_tokenizer=tokenizer,
            chat_template=chat_template,
        )
        self._orig_tokenizer = tokenizer

        super().__init__(
            tokenizer=my_tokenizer,
            compute_log_probs=compute_log_probs,
            enable_backtrack=enable_backtrack,
            enable_ff_tokens=enable_ff_tokens,
            enable_monitoring=enable_monitoring,
            **kwargs
        )

    def _model(self, model, **kwargs) -> "PreTrainedModel":
        # intantiate the model if needed
        if isinstance(model, str):

            # make sure transformers is installed
            if not has_transformers:
                raise Exception(
                    "Please install transformers with `pip install transformers` in order to use guidance.models.Transformers!"
                )
            model = transformers_package.AutoModelForCausalLM.from_pretrained(model, **kwargs)
        return model

    def get_logits(self, token_ids):
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
        max_cache_shape = None
        if past_key_values is None:
            past_length = 0
        elif isinstance(past_key_values, tuple):
            past_length = past_key_values[0][0].size(-2)
        elif isinstance(past_key_values, transformers_package.Cache):
            # TODO: use model's `cache_position` once available, as this may be deprecated in a future version
            # https://github.com/huggingface/transformers/blob/70b07d97cf2c5f61fff55700b65528a1b6845cd2/src/transformers/cache_utils.py#L64
            past_length = past_key_values.get_seq_length()
            try:
                max_cache_shape = past_key_values.get_max_cache_shape()
            except AttributeError:
                # `get_max_length` is deprecated in favor of `get_max_cache_shape`
                # as of transformers v4.48, but we use here for backwards compatibility
                max_cache_shape = past_key_values.get_max_length()
        else:
            raise TypeError(f"Unknown type of past_key_values: {type(past_key_values)}")

        if max_cache_shape is not None and len(token_ids) > max_cache_shape:
            # TODO: this seems to get set to the length of the first sequence we pass for models using
            # StaticCache or HybridCache. We need to initialize our own cache with a large enough size
            # if we want to continue generation with the same cache.
            if isinstance(
                past_key_values,
                (transformers_package.StaticCache, transformers_package.HybridCache),
            ):
                # The __init__ API isn't consistent between different cache types, but there seems to be consistency
                # between these two types, so we can use the same logic for both.
                warnings.warn("Cache is too small. Re-initializing cache with larger size.")
                cache_type = type(past_key_values)
                config = self.model_obj.config
                device = self.model_obj.device
                hf_device_map = getattr(self.model_obj, "hf_device_map", {})
                # hf_device_map is not always a complete mapping of layers to devices...
                layer_device_map = {
                    k: hf_device_map.get(k, device) for k in range(config.num_hidden_layers)
                }
                self._past_key_values = cache_type(
                    config=config,
                    max_batch_size=past_key_values.max_batch_size,
                    # Double the cache size to be safe
                    max_cache_len=len(token_ids) * 2,
                    dtype=past_key_values._dtype,
                    layer_device_map=layer_device_map,
                )
            else:
                warnings.warn(
                    f"Cache is too small. Resetting cache (no method implemented to resize cache for type {type(past_key_values)})."
                )
                self._past_key_values = None
            past_length = 0
        elif past_length > num_cached:
            past_length = max(0, num_cached - 1)
            if isinstance(past_key_values, tuple):
                self._past_key_values = tuple(
                    tuple(p[..., :past_length, :] for p in v) for v in past_key_values
                )
            else:
                if hasattr(past_key_values, "crop"):
                    self._past_key_values.crop(past_length)
                else:
                    warnings.warn(
                        f"Cropping unsupported for cache type: {type(self._past_key_values)}. Resetting cache."
                    )
                    if hasattr(self._past_key_values, "reset"):
                        # Use built-in reset method if available to avoid constructing/allocating a new cache
                        self._past_key_values.reset()
                    else:
                        self._past_key_values = None
                    past_length = 0

        cache_token_ids[past_length:] = []

        # call the model
        new_token_ids = token_ids[past_length:]
        if len(new_token_ids) > 0:
            with torch.no_grad():
                # Not all models support batched tokens for some reason
                try:
                    model_out = self.model_obj(
                        input_ids=torch.tensor(new_token_ids).unsqueeze(0).to(self.device),
                        past_key_values=self._past_key_values,
                        use_cache=True,
                        position_ids=torch.arange(past_length, past_length + len(new_token_ids))
                        .unsqueeze(0)
                        .to(self.device),
                        attention_mask=torch.ones(1, past_length + len(new_token_ids)).to(
                            self.device
                        ),
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                except AssertionError:
                    for i, new_token_id in enumerate(new_token_ids):
                        input_ids = torch.tensor([new_token_id]).unsqueeze(0).to(self.device)

                        model_out = self.model_obj(
                            input_ids=input_ids,
                            past_key_values=self._past_key_values,
                            use_cache=True,
                            position_ids=torch.arange(past_length, past_length + 1)
                            .unsqueeze(0)
                            .to(self.device),
                            attention_mask=torch.ones(1, past_length + 1).to(self.device),
                            return_dict=True,
                            output_attentions=False,
                            output_hidden_states=False,
                        )

                        self._past_key_values = model_out.past_key_values
                        past_length += 1

            # save the results
            self._past_key_values = model_out.past_key_values
            cache_token_ids.extend(new_token_ids)
            # Need to add special truncating logic here for weird models that have a different output size than tokenizer vocab
            self._cached_logits = (
                model_out.logits[0, -1, : self.tokenizer._vocab_size].float().cpu().numpy()
            )
            self.metrics.engine_input_tokens += len(new_token_ids)
            self.metrics.engine_output_tokens += 1

        return self._cached_logits

class Transformers(Model):
    def __init__(
        self,
        model: Union[str, "PreTrainedModel"],
        tokenizer: Union[
            "PreTrainedTokenizer",
            "PreTrainedTokenizerFast",
            None,
        ] = None,
        interpreter_cls: Optional[type[EngineInterpreter]] = None,
        echo=True,
        compute_log_probs=False,
        chat_template=None,
        enable_backtrack=True,
        enable_ff_tokens=True,
        enable_monitoring=True,
        **kwargs,
    ):
        """Build a new Transformers model object that represents a model in a given state."""
        if interpreter_cls is None and isinstance(model, str):
            if re.search("Llama-3.*-Vision", model):
                interpreter_cls = Llama3VisionInterpreter
            elif re.search("Phi-3-vision", model):
                interpreter_cls = Phi3VisionInterpreter
        if interpreter_cls is None:
                interpreter_cls = EngineInterpreter

        client = interpreter_cls(
            TransformersEngine(
                model,
                tokenizer,
                compute_log_probs,
                chat_template=chat_template,
                enable_backtrack=enable_backtrack,
                enable_ff_tokens=enable_ff_tokens,
                enable_monitoring=enable_monitoring,
                **kwargs,
            )
        )
        super().__init__(
            interpreter=client,
            echo=echo,
        )
