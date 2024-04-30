import requests
import os
import base64
import json
import urllib.parse
from ._model import Engine, Model, EngineCallResponse


class AzureGuidanceEngine(Engine):
    """This connects to a remote guidance server on Azure and runs all computation using the remote engine."""

    def __init__(self, server_url, max_streaming_tokens=1000):
        if (
            server_url is None
            or isinstance(server_url, str)
            and len(server_url.strip()) == 0
        ):
            server_url = os.getenv("AZURE_GUIDANCE_URL", "")
        elif not isinstance(server_url, str):
            raise ValueError("server_url must contain a URL string.")
        if not server_url.startswith("http"):
            raise ValueError(
                "AzureGuidance requires a remote model URL that starts with http"
            )
        self.server_url = server_url
        self.max_streaming_tokens = max_streaming_tokens

    def __call__(self, parser, grammar, ensure_bos_token=True):
        current_temp = 0.5  # TODO: handle temperature better

        b64 = base64.b64encode(grammar.serialize()).decode("utf-8")

        data = {
            "controller": "guidance_ctrl-latest",
            "controller_arg": json.dumps({"guidance_b64": b64}),
            "prompt": parser,
            "max_tokens": self.max_streaming_tokens,
            "temperature": current_temp,
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
            return RuntimeError(
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
                    texts = [""]
                    logs = [""]
                    capture_groups = {}
                    capture_group_log_probs = {}

                    if "Previous WASM Error" in ch["logs"]:
                        return RuntimeError("Previous WASM Error.")
                    idx = ch["index"]
                    while len(texts) <= idx:
                        texts.append("")
                        logs.append("")
                    for s in ch.get("storage", []):
                        w = s.get("WriteVar", None)
                        if w:
                            capture_groups[w["name"]] = w["value"]
                            capture_group_log_probs[w["name"]] = (
                                0  # TODO: get this from the server
                            )
                    err = ch.get("error", "")
                    if err:
                        return RuntimeError(f"Error returned by grammar server {err}.")
                    logs[idx] += ch["logs"]
                    texts[idx] += ch["text"]

                    # TODO: simplify this if it is always one
                    assert len(texts) == 1

                    new_bytes = bytes(texts[0], encoding="utf8")
                    is_generated = True  # TODO: get this from the server
                    new_bytes_prob = 1.0  # TODO: get this from the server
                    new_token_count = 1  # we process one token at a time

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
    ):
        """Build a new remote grammar processing Azure model object that represents a model in a given state."""

        engine = AzureGuidanceEngine(model, max_streaming_tokens)
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
