import requests
import os
import base64
import json
import urllib.parse
from ._model import Engine, Model, EngineCallResponse
from ..chat import Phi3MiniChatTemplate
from ._byte_tokenizer import ByteTokenizer


class AzureGuidanceEngine(Engine):
    """This connects to a remote guidance server on Azure and runs all computation using the remote engine."""

    def __init__(self, server_url, max_streaming_tokens=1000, chat_template=None):
        if (
            server_url is None
            or isinstance(server_url, str)
            and len(server_url.strip()) == 0
        ):
            server_url = os.getenv("AZURE_GUIDANCE_URL", "")
        elif not isinstance(server_url, str):
            raise ValueError("server_url must contain a URL string.")

        if not server_url.startswith("https://"):
            raise ValueError(
                "AzureGuidance requires a remote model URL that starts with https://"
            )
        self.server_url = server_url
        self.max_streaming_tokens = max_streaming_tokens

        if chat_template is None:
            # TODO [PK]: obtain this from the server
            chat_template=Phi3MiniChatTemplate

        tokenizer = ByteTokenizer(chat_template)

        # build the Engine
        super().__init__(tokenizer=tokenizer, compute_log_probs=False)

    def __call__(self, parser, grammar, ensure_bos_token=True):
        b64 = base64.b64encode(grammar.serialize()).decode("utf-8")

        data = {
            "controller": "guidance",
            "controller_arg": {"guidance_b64": b64},
            "prompt": parser,
            "max_tokens": self.max_streaming_tokens,
            "temperature": 0.0, # this is just default temperature
        }

        resp = req("post", "run", json=data, stream=True, base_url=self.server_url)
        if resp.status_code != 200:
            text = resp.text
            try:
                d = json.loads(text)
                if "message" in d:
                    text = d["message"]
            except:
                pass
            raise RuntimeError(
                f"Bad response to Guidance request {resp.status_code} {resp.reason}: {text}."
            )

        for line in resp.iter_lines():
            if not line:
                continue
            decoded_line: str = line.decode("utf-8")
            if decoded_line.startswith("data: {"):
                d = json.loads(decoded_line[6:])
                if "forks" not in d:
                    continue
                for ch in d["forks"]:
                    capture_groups = {}
                    capture_group_log_probs = {}

                    if "Previous WASM Error" in ch["logs"]:
                        raise RuntimeError("Previous WASM Error.")
                    idx = ch["index"]
                    assert idx == 0, "unexpected index in response from server"
                    new_bytes = b""
                    new_token_count = 0
                    new_bytes_prob = 0.0
                    num_text_entries = 0
                    for ln in ch["logs"].split("\n"):
                        ln: str
                        if ln.startswith("JSON-OUT: "):
                            j = json.loads(ln[10:])
                            tag = j.get("object", "")
                            if tag == "capture":
                                capture_groups[j["name"]] = bytes.fromhex(j["hex"])
                                capture_group_log_probs[j["name"]] = j["log_prob"]
                            elif tag == "text":
                                # it actually should only happen once per round...
                                new_bytes += bytes.fromhex(j["hex"])
                                new_token_count += j["num_tokens"]
                                new_bytes_prob += j["log_prob"]
                                num_text_entries += 1
                    if num_text_entries > 0:
                        new_bytes_prob /= num_text_entries

                    # print(ch["logs"].rstrip("\n"), flush=True)

                    err = ch.get("error", "")
                    if err:
                        raise RuntimeError(f"Error returned by grammar server {err}.")

                    is_generated = True  # TODO: get this from the server

                    response_data = EngineCallResponse(
                        new_bytes,
                        is_generated,
                        new_bytes_prob,
                        capture_groups,
                        capture_group_log_probs,
                        new_token_count,
                    )
                    yield response_data
            elif decoded_line == "data: [DONE]":
                pass
            else:
                raise RuntimeError(f"bad response line: {decoded_line}")


class AzureGuidance(Model):
    def __init__(
        self,
        model=None,
        echo=True,
        max_streaming_tokens=1000,
        chat_template=None,
    ):
        """Build a new remote grammar processing Azure model object that represents a model in a given state."""

        engine = AzureGuidanceEngine(model, max_streaming_tokens, chat_template)
        super().__init__(engine, echo=echo)


def _parse_base_url(base_url: str):
    p = urllib.parse.urlparse(base_url)
    key = ""
    if p.fragment:
        f = urllib.parse.parse_qs(p.fragment)
        key = f.get("key", [""])[0]
    r = urllib.parse.urlunparse(p._replace(fragment="", query=""))
    if not r.endswith("/"):
        r += "/"
    return r, key


def _headers(arg_base_url: str) -> dict:
    _, key = _parse_base_url(arg_base_url)
    if key:
        return {"api-key": key}
    else:
        return {}


def _mk_url(path: str, arg_base_url: str) -> str:
    pref, _ = _parse_base_url(arg_base_url)
    return pref + path


def req(tp: str, path: str, base_url: str, **kwargs):
    url = _mk_url(path, arg_base_url=base_url)
    headers = _headers(arg_base_url=base_url)
    resp = requests.request(tp, url, headers=headers, **kwargs)
    return resp
