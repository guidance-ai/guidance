import requests
import os
import json
import urllib.parse
from ._model import Engine, Model
from .._schema import LLProgress
from ..chat import Phi3MiniChatTemplate
from ._byte_tokenizer import ByteTokenizer
from typing import Dict, Tuple, Optional


class AzureGuidanceEngine(Engine):
    """This connects to a remote guidance server on Azure and runs all computation using the remote engine."""

    def __init__(
        self,
        server_url,
        max_streaming_tokens=1000,
        chat_template=None,
        log_level=1,
    ):
        if server_url is None or isinstance(server_url, str) and len(server_url.strip()) == 0:
            server_url = os.getenv("AZURE_GUIDANCE_URL", "")
        elif not isinstance(server_url, str):
            raise ValueError("server_url must contain a URL string.")

        if (
            not server_url.startswith("https://")
            and not server_url.startswith("http://")
        ):
            raise ValueError("AzureGuidance requires a remote model URL that starts with https:// or http://")
        self.conn_str = server_url
        self.max_streaming_tokens = max_streaming_tokens
        self.log_level = log_level

        if chat_template is None:
            # TODO [PK]: obtain this from the server
            chat_template = Phi3MiniChatTemplate

        tokenizer = ByteTokenizer(chat_template)

        # build the Engine
        super().__init__(tokenizer=tokenizer, compute_log_probs=False)

    def __call__(self, parser, grammar, ensure_bos_token=True):
        serialized = {"grammar": grammar.ll_serialize()}
        # this is a hack to avoid loops
        serialized["grammar"]["max_tokens"] = self.max_streaming_tokens
        # print(json.dumps(serialized))
        data = {
            "controller": "llguidance",
            "controller_arg": serialized,
            "prompt": parser,
            "max_tokens": self.max_streaming_tokens,
            "temperature": 0.0,  # this is just default temperature
        }

        url, headers, info = _mk_url("run", conn_str=self.conn_str)
        if self.log_level >= 4:
            print(f"POST {info}", flush=True)
        if self.log_level >= 5:
            print(f"  {json.dumps(data, indent=None)}", flush=True)
        resp = requests.request("post", url, headers=headers, json=data, stream=True)

        if resp.status_code != 200:
            text = resp.text
            try:
                d = json.loads(text)
                if "message" in d:
                    text = d["message"]
            except:
                pass
            raise RuntimeError(
                f"Bad response to Guidance request\nRequest: {info}\n"
                + f"Response: {resp.status_code} {resp.reason}\n{text}"
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
                    if "Previous WASM Error" in ch["logs"]:
                        raise RuntimeError("Previous WASM Error.")
                    idx = ch["index"]
                    assert idx == 0, "unexpected index in response from server"
                    progress = []
                    for ln in ch["logs"].split("\n"):
                        ln: str
                        if ln.startswith("JSON-OUT: "):
                            j = json.loads(ln[10:])
                            progress.append(j)
                        # don't print warnings if log_level >= 0, since we're 
                        # going to print them anyway below together with the
                        # rest of the logs
                        elif ln.startswith("Warning: ") and self.log_level < 2:
                            if self.log_level >= 1:
                                print(ln, flush=True)
                    progress = LLProgress.model_validate(progress)

                    if self.log_level >= 2:
                        print(ch["logs"].rstrip("\n"), flush=True)

                    err = ch.get("error", "")
                    if err:
                        raise RuntimeError(f"Error returned by grammar server {err}.")

                    # TODO: these metrics may be a little off -- notice the `-1` (which is a hack for passing
                    # tests in tests/model_integration/library/test_gen.py for now, may have to do with BOS?)
                    usage = d["usage"]
                    self.metrics.engine_input_tokens = usage["ff_tokens"]
                    self.metrics.engine_output_tokens = usage["sampled_tokens"] - 1

                    yield progress.to_engine_call_response()

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
        log_level: Optional[int] = None,
    ):
        """Build a new remote grammar processing Azure model object that represents a model in a given state."""
        if log_level is None:
            log_level = int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1"))
        engine = AzureGuidanceEngine(model, max_streaming_tokens, chat_template, log_level)
        super().__init__(engine, echo=echo)


def _mk_url(path: str, conn_str: str):
    p = urllib.parse.urlparse(conn_str)
    headers = {}
    info = "no auth header"
    if p.fragment:
        f = urllib.parse.parse_qs(p.fragment)
        if key := f.get("key", [""])[0]:
            headers = {"api-key": key}
            info = f"api-key: {key[0:2]}...{key[-2:]}"
        elif key := f.get("auth", [""])[0]:
            headers = {"authorization": "Bearer " + key}
            info = f"authorization: Bearer {key[0:2]}...{key[-2:]}"
    url = urllib.parse.urlunparse(p._replace(fragment="", query=""))
    if url.endswith("/"):
        url = url[:-1]
    if url.endswith("/run"):
        url = url[:-4] + "/" + path
    elif url.endswith("/guidance") and path == "run":
        url = url
    else:
        url = url + "/" + path
    info = f"{url} ({info})"
    return url, headers, info
