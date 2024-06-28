from typing import Any, List
import tokenizers
import llguidance
import json
from guidance import gen, select, optional, commit_point, byte_range, one_or_more

log_level = 1


class PhiTokenizer:
    _ll_tokenizer = None
    _instance = None

    @staticmethod
    def instance():
        if PhiTokenizer._instance is None:
            PhiTokenizer._instance = PhiTokenizer()
        return PhiTokenizer._instance

    @staticmethod
    def ll_tokenizer():
        if PhiTokenizer._ll_tokenizer is None:
            PhiTokenizer._ll_tokenizer = llguidance.LLTokenizer(
                llguidance.TokenizerWrapper(PhiTokenizer())
            )
        return PhiTokenizer._ll_tokenizer

    def tokenize_str(self, s: str) -> List[int]:
        return self.hf_tokenizer.encode(s).ids

    def __init__(self) -> None:
        self.hf_tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct"
        )
        empty = self.tokenize_str("")
        if empty:
            self.bos_token_id = empty[0]
        else:
            self.bos_token_id = None
        eos = self.tokenize_str("</s>")
        assert len(eos) == 1
        self.eos_token_id = eos[0]
        self.tokens = []
        for i in range(self.hf_tokenizer.get_vocab_size()):
            t: str = self.hf_tokenizer.id_to_token(i)
            if t.startswith("<0x"):
                self.tokens.append(bytes([int(t[3:5], 16)]))
            else:
                t = t.replace("▁", " ")
                self.tokens.append(t.encode("utf-8"))
        assert len(self.tokens) == self.hf_tokenizer.get_vocab_size()

    def __call__(self, s):
        return self.tokenize_str(s)


def check_eq(label: str, tokens: List[int], expected_tokens: str):
    if log_level > 0:
        print(f"Checking {label}: {repr(expected_tokens)}")
    t = PhiTokenizer.ll_tokenizer()
    actual_tokens = t.test_trace_tokens(tokens)
    assert (
        actual_tokens == expected_tokens
    ), f"Tokens mismatch in {label}\n  {repr(actual_tokens)}\n  {repr(expected_tokens)}"


def tokenize_trace(s: str):
    if log_level > 0:
        print("Tokenizing", repr(s))
    r: List[int] = []
    for word in s.split("‧"):
        if word == '≺EOS≻':
            r.append(PhiTokenizer.instance().eos_token_id)
            continue
        tt = PhiTokenizer.ll_tokenizer().tokenize_str(word)
        assert len(tt) == 1, f"Expected single token for {repr(word)} got {tt}"
        r.append(tt[0])
    return r


def check_grammar(grm, output: List[str]):
    """
    Check that the grammar generates the expected output.

    Output is a list of strings, each of which is a sequence of tokens.
    Tokens in the string are separated with "‧".
    Strings at even positions are "forced tokens", and strings at odd positions
    are "generated tokens".
    We check that the grammars forces the forced tokens (first of which is the
    prompt), and that it allows in the mask the generated tokens.

    These tests are "recorded" by passing "test_trace": true in the llguidance
    request and post-processing.
    """
    print("\nChecking grammar")
    interp = llguidance.LLInterpreter(
        PhiTokenizer.ll_tokenizer(), json.dumps(grm.ll_serialize()), log_level=log_level
    )
    prompt = interp.process_prompt(PhiTokenizer.instance().tokenize_str(""))
    check_eq("prompt", prompt, output[0])
    idx = 1
    bt = 0
    toks: List[int] = []
    gen_tokens = tokenize_trace(output[idx])
    for _ in range(200):
        mask, cmd = interp.mid_process(bt, toks)
        cmd = json.loads(cmd)
        if cmd["stop"]:
            assert idx >= len(output) - 1, f"Expected more output at {idx}"
            assert not gen_tokens, "Expected more tokens to generate"
            break
        if mask:
            if not gen_tokens:
                raise ValueError("No more tokens to generate")
            tok = gen_tokens[0]
            del gen_tokens[0:1]
            assert mask[tok] > 0, f"Token {tok} not allowed"
            toks = [tok]
            bt = 0
        else:
            bt = cmd["backtrack"]
            toks = cmd["ff_tokens"]
            assert not gen_tokens, "Expected more tokens to generate"
            idx += 1
            check_eq(f"step {idx}", toks, output[idx])
            idx += 1
            if idx < len(output):
                gen_tokens = tokenize_trace(output[idx])


def test_llparser():
    grm = (
        "Q: Are dolphins fish?\nA: "
        + gen("dolphins", regex="Yes|No", max_tokens=10)
        + "\nQ: Are sharks fish?\nA: "
        + gen("sharks", regex="Yes|No", max_tokens=10)
    )
    check_grammar(
        grm,
        [
            "Q‧:‧ Are‧ dol‧ph‧ins‧ fish‧?‧\n‧A‧:",
            " No",
            "\n‧Q‧:‧ Are‧ sh‧arks‧ fish‧?‧\n‧A‧:",
            " Yes",
        ],
    )

    grm = (
        "Power frequency is "
        + gen("number", regex="[0-9]+", max_tokens=5)
        + "Hz; voltage is "
        + gen("number", regex="[0-9]+", max_tokens=5)
        + "V"
    )
    check_grammar(grm, ["Power‧ frequency‧ is‧ ", "5‧0‧Hz", ";‧ voltage‧ is‧ ", "2‧2‧0‧V"])

    grm = "Q: 7 * 8\nA: " + gen("text", regex="[0-9]+", max_tokens=5)
    check_grammar(grm, ["Q‧:‧ ‧7‧ *‧ ‧8‧\n‧A‧:‧ ", "5‧6‧≺EOS≻"])

    grm = "Dolphin name: " + commit_point(
        '"' + byte_range(b"A", b"Z") + one_or_more(byte_range(b"a", b"z")) + '"'
    ) + ","
    check_grammar(grm, ['D‧olph‧in‧ name‧:‧ "', 'F‧li‧pper‧"', ','])


if __name__ == "__main__":
    test_llparser()
