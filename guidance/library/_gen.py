import types
import regex

import guidance
import ast

@guidance
def gen(lm, name=None, *, max_tokens=1000, list_append=False, pattern=None, stop=None, stop_regex=None, suffix="", n=1, temperature=0.0, top_p=1.0,
        logprobs=None, cache_seed=None, token_healing=None, stream=None, function_call="none", save_stop_text=False, **llm_kwargs):

    # set stream if we are interactive
    if stream is None and not lm.silent:
        stream = True

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

    lm += "<||_html:<span style='background-color: rgba(0, 165, 0, 0.25)'>_||>"

    # This needs to be here for streaming
    if name is not None and not list_append:
        lm[name] = ""

    gen_obj = lm.get_endpoint_session()(
        str(lm), stop=stop, stop_regex=stop_regex, max_tokens=max_tokens, n=n, pattern=pattern,
        temperature=temperature, top_p=top_p, logprobs=logprobs, cache_seed=cache_seed, token_healing=token_healing,
        echo=getattr(lm, "logprobs", False), stream=stream, function_call=function_call, **llm_kwargs
    )

    if not isinstance(gen_obj, (types.GeneratorType, list, tuple)):
        gen_obj = [gen_obj]

    if n == 1:
        generated_value = ""
        logprobs_out = []
        if list_append:
            lm[name] = lm.get(name, [])
            lm[name].append("")
            list_ind = len(lm[name])-1
            if logprobs is not None:
                lm[name+"_logprobs"] = lm.get(name+"_logprobs", [])
                lm[name+"_logprobs"].append([])
                assert len(len(lm[name])) == len(len(lm[name+"_logprobs"]))
        for resp in gen_obj:
            new_text = resp["choices"][0].get("text", "")
            generated_value += new_text
            lm += new_text
            if logprobs is not None:
                logprobs_out.extend(resp["choices"][0]["logprobs"]["top_logprobs"])
            if list_append:
                lm[name][list_ind] = generated_value
                if logprobs is not None:
                    lm[name+"_logprobs"][list_ind] = logprobs_out
            elif name is not None:
                lm[name] = generated_value
                if logprobs is not None:
                    lm[name+"_logprobs"] = logprobs_out
        
        # save the final stopping text if requested
        if save_stop_text is not False:
            if save_stop_text is True and name is not None:
                save_stop_text = name+"_stop_text"
            lm[save_stop_text] = resp["choices"][0].get('stop_text', None)
        
        # for scanner in lm.get_call_scanners():
        #     out = scanner(lm, generated_value)
        #     if out is not None:
        #         generated_value = out

        if list_append:
            lm[name][list_ind] = generated_value
        elif name is not None:
            # This seems wrong, it's overriding whatever was generated into the name. What if the generation was 'I am now going to call a tool: tool_call(bla)', do you want to just dump that?
            lm[name] = generated_value
    
    lm += "<||_html:</span>_||>" + suffix
    
    return lm

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
    optional_space = '\s*' if ignore_spaces else ''
    pattern = regex.compile(f'{optional_space}({"|".join(regexes)})')
    with lm.block(hidden=True):
        for _ in range(max_tokens):
            lm.gen('temp_variable', list_append=True, max_tokens=1)
            print(lm['temp_variable'])
            if not pattern.match(''.join(lm['temp_variable']), partial=True):
                lm.remove('temp_variable')
                return False
            if pattern.match(''.join(lm['temp_variable']), partial=False):
                lm.remove('temp_variable')
                return True

@guidance
def gen_substring(lm, string, name=None, **kwargs):
    # this is obviously not the right implementation, just here so we can explore
    # Right now it's kinda wrong because it does not support suffix (or other kwargs to gen),
    # which is a big pain because the model ends up being bad at stopping
    # (will never pick stop token until the string is finished, unless you have the suffix as an option also)
    # gen_substring will never stop unless you add a suffix as an option or set max_tokens or stop
    # E.g. quote: lm('"').gen_substring(original, suffix='"') 

    tokens = [lm.endpoint.tokenizer.decode(x) for x in lm.endpoint.tokenizer.encode(string)]
    tokens += [x.strip() for x in tokens if x.strip() != x]
    tokens = [x for x in tokens if x]
    pattern = f'({"|".join(tokens)})?'
    if name is None:
        name = 'temp_string'
        remove_temp = True
    # return tokens, pattern
    lm.gen('temp_string', pattern=pattern)
    valid_idxs = [i for i, x in enumerate(tokens) if x == lm['temp_string'] ]
    while valid_idxs:
        next_idxs = [i + 1 for i in valid_idxs if i + 1 < len(tokens)]
        next_tokens = [tokens[i] for i in next_idxs]
        if not next_tokens:
            break
        pattern = f'({"|".join(next_tokens)})?'
        lm.gen('temp_string', pattern=pattern)
        valid_idxs = [i for i in next_idxs if tokens[i] == lm['temp_string']]
    if remove_temp:
        lm.remove('temp_string')
    return lm


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
def gen_with_tools(lm, name=None, tools=None, stop_on_tool=False, **kwargs):
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
        pattern = f'{name}\((.*)\)'
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
    while called_tool:
        called_tool = False
        lm.gen(name='temp_name', **kwargs)
        temp_output += lm['temp_name']
        if lm['temp_name_stop_text'] is None:
            break
        for p in to_callables:
            ret = p(lm['temp_name_stop_text'])
            if ret is not None:
                callable, targs, tkwargs = ret
                temp_output += lm['temp_name_stop_text']
                lm.append(lm['temp_name_stop_text'])
                callable(lm, *targs, **tkwargs)
                called_tool = True if not stop_on_tool else False
                break
    if gen_name is not None:
        lm[gen_name] = temp_output
    return lm

@guidance
def call_tool(lm, tool):
    name = tool.__name__
    pattern = f'{name}\(([^)]*)\)'
    p_to_callable = pattern_to_callable(pattern, tool)
    lm.gen('fn_call', pattern=pattern)
    callable, args, kwargs = p_to_callable(lm['fn_call'])
    callable(lm, *args, **kwargs)
    return lm
