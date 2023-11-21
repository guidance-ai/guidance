import types
import regex as lregex
import uuid

import guidance
import ast

# from guidance import select, any_char, zero_or_more, commit_point, hide
from ._silent import silent
from .._grammar import select
from ._zero_or_more import zero_or_more
from .._grammar import commit_point
from ._any_char import any_char
from .._grammar import capture
from ._regex import regex as regex_grammar
from .._grammar import token_limit
from .._grammar import with_temperature
from .._grammar import model_variable
from ._tool import Tool

# TODO: make this stateless!
@guidance(stateless=lambda *args, **kwargs: kwargs.get("tools", None) is None) # TODO: uncomment this once we get temperature stateless
def gen(lm, name=None, *, max_tokens=1000, list_append=False, regex=None,
        tools=None, hide_tool_call=False, stop=None, stop_regex=None, suffix="", n=1, temperature=0.0, top_p=1.0,
        logprobs=None, stream_tokens=None, save_stop_text=False, **llm_kwargs):
    """
    TODO: document this
    tools is a list of guidance.Tool or python functions (which will be converted to guidance.Tool)
    """

    # set stream if we are interactive
    # if stream_tokens is None and not lm.is_silent() and n == 1:
    #     stream_tokens = True

    # use the suffix as the stop string if not otherwise specified
    # TODO: still need to make suffix work with grammars
    # eos_token = lm.eos_token.decode('utf8')
    if stop is None and stop_regex is None and suffix != "":
        stop = suffix
    # if stop is None and stop_regex is None and getattr(lm, "suffix", False):
    #     if lm.suffix.startswith("\n"):
    #         stop = "\n"
    #     elif lm.suffix.startswith('"') and str(lm).endswith('"'):
    #         stop = '"'
    #     elif lm.suffix.startswith("'") and str(lm).endswith("'"):
    #         stop = "'"

    # fall back to stopping at the EOS token
    if stop is None:
        stop = []
    if isinstance(stop, str):
        stop = [stop]
    if regex is None:
        stop.append(model_variable('default_end_patterns'))

    if stop_regex is None:
        stop_regex = []
    if isinstance(stop_regex, str):
        stop_regex = [stop_regex]
    stop_regex = [regex_grammar(x) for x in stop_regex]

    # This needs to be here for streaming
    # if name is not None and not list_append:
    #     lm[name] = ""

    # TODO: This can be uncommented once we support regex -> grammar compilation
    
    # compile stop_regex into a capture group
    # if "(?P<stop>" in pattern:
    #     assert stop_regex == None, "You can't specify both a stop string/regex and a custom stop pattern (?P<stop>...)!"
    #     stop_regex = ""
    # else:
    #     if stop_regex is None:
    #         stop_regex = []
    #     stop_regex.append(regex.escape(lm.eos_token))
    #     stop_regex = "(?P<stop>" + "|".join(stop_regex) + ")"

    # pattern += stop_regex
    # extracted_stop_pattern = regex.compile(pattern[pattern.index("(?P<stop>")+9:-1] + "$", flags=regex.DOTALL)
    # extracted_stop_pattern = regex.compile(pattern[:pattern.index("(?P<stop>")] + "$", flags=regex.DOTALL)
    
    # define the generation pattern
    if regex is not None:
        pattern = regex_grammar(regex)
    else:
        pattern = zero_or_more(any_char())

    # define any capture group
    if name is not None:
        pattern = capture(pattern, name="__LIST_APPEND:" + name if list_append else name)
    
    # limit the number of tokens
    pattern = token_limit(pattern, max_tokens)
    
    # define the stop pattern
    if stop + stop_regex:
        stop_pattern = select(stop + stop_regex)
        if save_stop_text is True:
            save_stop_text = str(name) + "_stop_text"
        if isinstance(save_stop_text, str):
            stop_pattern = capture(stop_pattern, name=save_stop_text)
        stop_pattern = commit_point(stop_pattern, hidden=True)
    else:
        stop_pattern = ''

    # single generation
    start_pos = len(str(lm))
    if tools is not None:
        tools = [Tool(callable=x) if not isinstance(x, Tool) else x for x in tools]
        init_token_count = lm.token_count
        gen_grammar = pattern + select([stop_pattern] + [capture(commit_point(x.call_grammar, hidden=hide_tool_call), name=f'tool{i}') for i, x in enumerate(tools)])
        while lm.token_count <= max_tokens + init_token_count:
            lm = lm._run_stateless(gen_grammar, temperature=temperature) # TODO: we should not be using this internal method
            tool_called = False
            for i in range(len(tools)):
                tool_i = f'tool{i}'
                if tool_i in lm:
                    tool_called = True
                    lm += tools[i].tool_call()
                    lm = lm.remove(tool_i)
            if not tool_called:
                lm += suffix
                break
    elif n == 1:
        lm += with_temperature(pattern + stop_pattern + suffix, temperature)

    return lm


def click_loop_start(id, total_count, echo, color):
    click_script = '''
function cycle_IDVAL(button_el) {
var i = 0;
while (i < 50) {
var el = document.getElementById("IDVAL_" + i);
if (el.style.display == "inline") {
    el.style.display = "none";
    var next_el = document.getElementById("IDVAL_" + (i+1));
    if (!next_el) {
        next_el = document.getElementById("IDVAL_0");
    }
    if (next_el) {
        next_el.style.display = "inline";
    }
    break;
}
i += 1;
}
button_el.innerHTML = (((i+1) % TOTALCOUNT) + 1)  + "/" + TOTALCOUNT;
}
cycle_IDVAL(this);'''.replace("IDVAL", id).replace("TOTALCOUNT", str(total_count)).replace("\n", "")
    out = f'''<div style='background: rgba(255, 255, 255, 0.0); border-radius: 4px 0px 0px 4px; border: 1px solid {color}; border-right: 0px; padding-left: 3px; padding-right: 3px; user-select: none; color: {color}; display: inline; font-weight: normal; cursor: pointer' onClick='{click_script}'>1/{total_count}</div>'''
    out += f"<div style='display: inline;' id='{id}_0'>"
    return "<||_html:" + out + "_||>"

def click_loop_mid(id, index, echo):
    alpha = 1.0 if not echo else 0.5
    out = f"</div><div style='display: none; opacity: {alpha}' id='{id}_{index}'>"
    return "<||_html:" + out + "_||>"

@guidance
def gen_line(lm, *args, **kwargs):
    return lm.gen(*args, suffix='\n', **kwargs)

@guidance
def gen_quote(lm, name=None, quote='"', *args, **kwargs):
    return lm(quote).gen(*args,name=name, suffix=quote, **kwargs)

@guidance
def will_gen(lm, stop=None, stop_regex=None, ignore_spaces=False, max_tokens=30):
    # this is obviously not the right implementation, just here so we can explore
    if stop and not isinstance(stop, list):
        stop = [stop]
    if stop_regex and not isinstance(stop_regex, list):
        stop_regex = [stop_regex]
    assert (stop is not None) or (stop_regex is not None)
    if not stop:
        stop = []
    if not stop_regex:
        stop_regex = []
    regexes = [lregex.escape(x) for x in stop + stop_regex]
    optional_space = '\\s*' if ignore_spaces else ''
    pattern = lregex.compile(f'{optional_space}({"|".join(regexes)})')
    lm2 = lm
    with silent():
        for _ in range(max_tokens):
            lm2 += gen('temp_variable', list_append=True, max_tokens=1)
            if not lm2['temp_variable'] or not pattern.match(''.join(lm2['temp_variable']), partial=True):
                return False
            if pattern.match(''.join(lm2['temp_variable']), partial=False):
                return True
    return False

@guidance
def call_tool(lm, tool):
    return lm + tool.call_grammar + tool.tool_call()