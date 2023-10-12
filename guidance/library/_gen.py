import types
import regex
import uuid

import guidance
import ast

# from guidance import select, any_char, zero_or_more, commit_point, hide
from ._select import select
from ._zero_or_more import zero_or_more
from ._hide import hide
from ._commit_point import commit_point
from ._any_char import any_char
from ._capture import capture

@guidance
def gen(lm, name=None, *, max_tokens=1000, list_append=False, pattern=None, stop=None, stop_regex=None, suffix="", n=1, temperature=0.0, top_p=1.0,
        logprobs=None, stream_tokens=None, save_stop_text=False, **llm_kwargs):
    
    assert pattern is None, "Need to support regex -> grammar compilation in order to support the `pattern` arg!"
    assert stop_regex is None, "Need to support regex -> grammar compilation in order to support the `stop_regex` arg!"

    # set stream if we are interactive
    # if stream_tokens is None and not lm.is_silent() and n == 1:
    #     stream_tokens = True

    # use the suffix as the stop string if not otherwise specified
    if stop is None and stop_regex is None and suffix != "":
        stop = suffix
    if stop is None and stop_regex is None and getattr(lm, "suffix", False):
        if lm.suffix.startswith("\n"):
            stop = "\n"
        elif lm.suffix.startswith('"') and lm.endswith('"'):
            stop = '"'
        elif lm.suffix.startswith("'") and lm.endswith("'"):
            stop = "'"

    # fall back to stopping at the EOS token
    if stop is None:
        stop = lm.eos_token.decode("utf8")

    # standardize stop and stop_regex into a list of regex patterns
    if isinstance(stop, str):
        stop = [stop]
    # TODO: This can be uncommented once we support regex -> grammar compilation
    # if isinstance(stop_regex, str):
    #     stop_regex = [stop_regex]
    # if stop is not None:
    #     if stop_regex is None:
    #         stop_regex = []
    #     stop_regex.extend([regex.escape(s) for s in stop])

    # This needs to be here for streaming
    if name is not None and not list_append:
        lm[name] = ""

    # TODO: This can be uncommented once we support regex -> grammar compilation
    # if pattern is None:
    #     pattern = ".*"
    
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
    pattern = zero_or_more(any_char())
    if name is not None:
        pattern = capture(pattern, name=name)
    
    # define the stop pattern
    stop_pattern = commit_point(select(stop))
    if save_stop_text is True:
        save_stop_text = str(name) + "_stop_text"
    if isinstance(save_stop_text, str):
        stop_pattern = capture(save_stop_text, name=save_stop_text)

    # return the compound grammar
    return lm + (pattern + hide(stop_pattern))

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
    regexes = [regex.escape(x) for x in stop + stop_regex]
    optional_space = '\\s*' if ignore_spaces else ''
    pattern = regex.compile(f'{optional_space}({"|".join(regexes)})')
    with lm.silent() as lm2:
        for _ in range(max_tokens):
            lm2 += gen('temp_variable', list_append=True, max_tokens=1)
            if not pattern.match(''.join(lm2['temp_variable']), partial=True):
                return False
            if pattern.match(''.join(lm2['temp_variable']), partial=False):
                return True
    return False
    # with lm.block(hidden=True):
    #     for _ in range(max_tokens):
    #         lm.gen('temp_variable', list_append=True, max_tokens=1)
    #         if not pattern.match(''.join(lm['temp_variable']), partial=True):
    #             lm.remove('temp_variable')
    #             return False
    #         if pattern.match(''.join(lm['temp_variable']), partial=False):
    #             lm.remove('temp_variable')
    #             return True

@guidance
def gen_substring(lm, string, name=None, **kwargs):
    # this is obviously not the right implementation, just here so we can explore
    # Right now it's kinda wrong because it does not support suffix (or other kwargs to gen),
    # which is a big pain because the model ends up being bad at stopping
    # (will never pick stop token until the string is finished, unless you have the suffix as an option also)
    # gen_substring will never stop unless you add a suffix as an option or set max_tokens or stop
    # E.g. quote: lm('"').gen_substring(original, suffix='"') 

    try:
        tokenized = lm.endpoint.tokenizer(string, return_offsets_mapping=True)
        tokens = [string[a[0]:a[1]] for a in tokenized['offset_mapping']]
        tokens = [x for x in tokens if x]
        stripped = tokens + [x.strip() for x in tokens if x.strip() != x]
        stripped = [x for x in stripped if x]
    except:
        tokens = [lm.endpoint.tokenizer.decode(x) for x in lm.endpoint.tokenizer.encode(string)]
        tokens = [x for x in tokens if x]
        stripped = tokens + [x.strip() for x in tokens if x.strip() != x]
        stripped = [x for x in stripped if x]
    pattern = f'({"|".join(stripped)})?'
    if name is None:
        name = 'temp_string'
    # return tokens, pattern
    lm2 = lm
    lm2 += gen('temp_string', pattern=pattern, **kwargs)
    valid_idxs = [i for i, x in enumerate(tokens) if x == lm2['temp_string'] ]
    while valid_idxs:
        next_idxs = [i + 1 for i in valid_idxs if i + 1 < len(tokens)]
        next_tokens = [tokens[i] for i in next_idxs]
        if not next_tokens:
            break
        pattern = f'({"|".join(next_tokens)})?'
        lm2 += gen('temp_string', pattern=pattern, **kwargs)
        valid_idxs = [i for i in next_idxs if tokens[i] == lm2['temp_string']]

    list_append = kwargs.get('list_append', False)
    if list_append:
        prev_list = lm2.get(name, [])
        prev_list.append(str(lm2)[len(str(lm)):])
        lm2 = lm2.set(name, prev_list)
    else:
        lm2 = lm2.set(name, str(lm2)[len(str(lm)):])
    return lm2


def pattern_to_callable(pattern, callable):
    # returns callable, args, kwargs
    pattern = regex.compile(pattern)
    def return_fn(string):
        match = pattern.search(string)
        if match:
            call = match.group(0)
            # TODO: Remove this
            try:
                body = ast.parse(call, mode='eval').body
            except:
                return None
            args = [x.value for x in body.args]
            kwargs = {x.arg: x.value.value for x in body.keywords}
            return callable, args, kwargs
        return None
    return return_fn

@guidance
def gen_with_tools(lm, name=None, tools=None, stop_on_tool=False, include_tool_call=True, **kwargs):
    # V0 to see if this interface is good:
    # tools is a list of guidance functions.
    # In this v0, we only support python function calls, where the pattern is fn_name(args).
    # Not keeping track of maxtokens.
    # What this is doing:
    # 1. call gen with tool patterns as stop fns
    # 2. when gen stops, see if a tool stopped it. If so, call the tool, then call gen again until gen returns due to other stuff.
    # NOTE: This doesn't work for nested expressions, e.g. ((3 * 3) + 1).... or with any arguments that have params, unfortunately
    patterns = []
    to_callables = []
    gen_name = name
    for tool in tools:
        name = tool.__name__
        pattern = f'{name}\\((.*)\\)'
        patterns.append(pattern)
        p_to_callable = pattern_to_callable(pattern, tool)
        to_callables.append(p_to_callable)
    # return patterns
    if 'stop_regex' in kwargs:
        if isistance(kwargs['stop_regex'], list):
            kwargs['stop_regex'].extend(patterns)
        else:
            kwargs['stop_regex']  = [kwargs['stop_regex']] + patterns
    else:
        kwargs['stop_regex'] = patterns
    kwargs['save_stop_text'] = True
    called_tool = True
    temp_output = ''
    new = lm
    tool_calls = []
    while called_tool:
        called_tool = False
        new += gen(name='temp_name', **kwargs)
        if new['temp_name_stop_text'] is None:
            break
        for p in to_callables:
            tool_call = new['temp_name_stop_text']
            tool_calls.append(tool_call)
            ret = p(tool_call)
            if ret is not None:
                callable, targs, tkwargs = ret
                if include_tool_call:
                    new += tool_call
                new += callable(*targs, *tkwargs)
                called_tool = True if not stop_on_tool else False
                break
    list_append = kwargs.get('list_append', False)
    if list_append:
        tc = new.get('tool_calls', [])
        tc.append(tool_calls)
        new = new.set('tool_calls', tc)
        prev_list = new.get(gen_name, [])
        prev_list.append(str(new)[len(str(lm)):])
        new = new.set(gen_name, prev_list)
    else:
        new = new.set('tool_calls', tool_calls)
        new = new.set(gen_name, str(new)[len(str(lm)):])
    return new

@guidance
def call_tool(lm, tool):
    name = tool.__name__
    pattern = f'{name}\\(([^)]*)\\)'
    p_to_callable = pattern_to_callable(pattern, tool)
    lm += gen('fn_call', pattern=pattern)
    callable, args, kwargs = p_to_callable(lm['fn_call'])
    lm += callable(*args, **kwargs)
    return lm
