import asyncio
import re
import uuid
import logging
import types
from .._utils import escape_template_block, AsyncIter

log = logging.getLogger(__name__)

async def gen(name=None, stop=None, stop_regex=None, save_stop_text=False, max_tokens=500, n=1, stream=None,
              temperature=0.0, top_p=1.0, logprobs=None, pattern=None, hidden=False, list_append=False,
              save_prompt=False, token_healing=None, function_call="none", _parser_context=None, **llm_kwargs):
    ''' Use the LLM to generate a completion.

    Parameters
    ----------
    name : str or None
        The name of a variable to store the generated value in. If none the value is just returned.
    stop : str
        The stop string to use for stopping generation. If not provided, the next node's text will be used if
        that text matches a closing quote, XML tag, or role end. Note that the stop string is not included in
        the generated value.
    stop_regex : str
        A regular expression to use for stopping generation. If not provided, the stop string will be used.
    save_stop_text : str or bool
        If set to a string, the exact stop text used will be saved in a variable with the given name. If set to
        True, the stop text will be saved in a variable named `name+"_stop_text"`. If set to False,
        the stop text will not be saved.
    max_tokens : int
        The maximum number of tokens to generate in this completion.
    n : int
        The number of completions to generate. If you generate more than one completion, the variable will be
        set to a list of generated values. Only the first completion will be used for future context for the LLM,
        but you may often want to use hidden=True when using n > 1.
    temperature : float
        The temperature to use for generation. A higher temperature will result in more random completions. Note
        that caching is always on for temperature=0, and is seed-based for other temperatures.
    top_p : float
        The top_p value to use for generation. A higher top_p will result in more random completions.
    logprobs : int or None
        If set to an integer, the LLM will return that number of top log probabilities for the generated tokens
        which will be stored in a variable named `name+"_logprobs"`. If set to None, the log
        probabilities will not be returned.
    pattern : str or None
        A regular expression pattern guide to use for generation. If set the LLM will be forced (through guided
        decoding) to only generate completions that match the regular expression.
    hidden : bool
        Whether to hide the generated value from future LLM context. This is useful for generating completions
        that you just want to save in a variable and not use for future context.
    list_append : bool
        Whether to append the generated value to a list stored in the variable. If set to True, the variable
        must be a list, and the generated value will be appended to the list.
    save_prompt : str or bool
        If set to a string, the exact prompt given to the LLM will be saved in a variable with the given name.
    token_healing : bool or None
        If set to a bool this overrides the token_healing setting for the LLM.
    **llm_kwargs
        Any other keyword arguments will be passed to the LLM __call__ method. This can be useful for setting
        LLM-specific parameters like `repetition_penalty` for Transformers models or `suffix` for some OpenAI models.
    '''
    prefix = ""
    suffix = ""

    # get the parser context variables we will need
    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']
    next_node = _parser_context["next_node"]
    next_next_node = _parser_context["next_next_node"]
    prev_node = _parser_context["prev_node"]
    # partial_output = _parser_context["partial_output"]
    pos = len(variable_stack["@raw_prefix"]) # save the current position in the prefix

    if hidden:
        variable_stack.push({"@raw_prefix": variable_stack["@raw_prefix"]})

    if list_append:
        assert name is not None, "You must provide a variable name when using list_append=True"

    # if stop is None then we use the text of the node after the generate command
    if stop is None:

        next_text = getattr(next_node, "text", next_node) if next_node is not None else ""
        prev_text = getattr(prev_node, "text", prev_node) if prev_node is not None else ""
        if next_next_node and next_text == "":
            next_text = getattr(next_next_node, "text", next_next_node)

        # auto-detect quote stop tokens
        quote_types = ["'''", '"""', '```', '"', "'", "`"]
        for quote_type in quote_types:
            if next_text.startswith(quote_type) and prev_text.endswith(quote_type):
                stop = quote_type
                break

        # auto-detect role stop tags
        if stop is None:
            m = re.match(r"^{{~?/\w*(user|assistant|system|role|function)\w*~?}}.*", next_text)
            if m:
                stop = parser.program.llm.role_end(m.group(1))

        # auto-detect XML tag stop tokens
        if stop is None:
            m = re.match(r"<([^>\W]+)[^>]+>", next_text)
            if m is not None:
                end_tag = "</"+m.group(1)+">"
                if next_text.startswith(end_tag):
                    stop = end_tag
        
        # fall back to the next node's text (this was too easy to accidentally trigger, so we disable it now)
        # if stop is None:
        #     stop = next_text

    if stop == "":
        stop = None

    # set the cache seed to 0 if temperature is 0
    if temperature > 0:
        cache_seed = parser.program.cache_seed
        parser.program.cache_seed += 1
    else:
        cache_seed = 0

    # set streaming default
    if stream is None:
        stream = parser.program.stream or parser.program._displaying or stop_regex is not None if n == 1 else False

    # we can't stream batches right now TODO: fix this
    assert not (stream and n > 1), "You can't stream batches of completions right now."
    # stream_generation = parser.program.stream if n == 1 else False

    # save the prompt if requested
    if save_prompt:
        variable_stack[save_prompt] = variable_stack["@raw_prefix"]+prefix

    if logprobs is None:
        logprobs = parser.program.logprobs

    assert parser.llm_session is not None, "You must set an LLM for the program to use (use the `llm=` parameter) before you can use the `gen` command."

    # call the LLM
    gen_obj = await parser.llm_session(
        variable_stack["@prefix"]+prefix, stop=stop, stop_regex=stop_regex, max_tokens=max_tokens, n=n, pattern=pattern,
        temperature=temperature, top_p=top_p, logprobs=logprobs, cache_seed=cache_seed, token_healing=token_healing,
        echo=parser.program.logprobs is not None, stream=stream, caching=parser.program.caching, function_call=function_call, **llm_kwargs
    )

    if n == 1:
        generated_value = prefix
        variable_stack["@raw_prefix"] += prefix
        logprobs_out = []
        if not isinstance(gen_obj, (types.AsyncGeneratorType, types.GeneratorType, list, tuple)):
            gen_obj = [gen_obj]
        if not isinstance(gen_obj, types.AsyncGeneratorType):
            gen_obj = AsyncIter(gen_obj)
        if list_append:
            variable_stack[name] = variable_stack.get(name, [])
            variable_stack[name].append("")
            list_ind = len(variable_stack[name])-1
            if logprobs is not None:
                variable_stack[name+"_logprobs"] = variable_stack.get(name+"_logprobs", [])
                variable_stack[name+"_logprobs"].append([])
                assert len(len(variable_stack[name])) == len(len(variable_stack[name+"_logprobs"]))
        async for resp in gen_obj:
            await asyncio.sleep(0) # allow other tasks to run
            #log("parser.should_stop = " + str(parser.should_stop))
            if parser.should_stop:
                #log("Stopping generation")
                break
            # log.debug("resp", resp)
            new_text = resp["choices"][0].get("text", "")
            generated_value += new_text
            variable_stack["@raw_prefix"] += new_text
            if logprobs is not None:
                logprobs_out.extend(resp["choices"][0]["logprobs"]["top_logprobs"])
            if list_append:
                variable_stack[name][list_ind] = generated_value
                if logprobs is not None:
                    variable_stack[name+"_logprobs"][list_ind] = logprobs_out
            elif name is not None:
                variable_stack[name] = generated_value
                if logprobs is not None:
                    variable_stack[name+"_logprobs"] = logprobs_out
        
        # save the final stopping text if requested
        if save_stop_text is not False:
            if save_stop_text is True:
                save_stop_text = name+"_stop_text"
            variable_stack[save_stop_text] = resp["choices"][0].get('stop_text', None)
        
        if hasattr(gen_obj, 'close'):
            gen_obj.close()
        generated_value += suffix
        variable_stack["@raw_prefix"] += suffix
        if list_append:
            variable_stack[name][list_ind] = generated_value
        elif name is not None:
            variable_stack[name] = generated_value

        if hidden:
            new_content = variable_stack["@raw_prefix"][pos:]
            variable_stack.pop()
            variable_stack["@raw_prefix"] += "{{!--GHIDDEN:"+new_content.replace("--}}", "--_END_END")+"--}}"
        
        # stop executing if we were interrupted
        if parser.should_stop:
            parser.executing = False
            parser.should_stop = False
        return
    else:
        assert not isinstance(gen_obj, list), "Streaming is only supported for n=1"
        generated_values = [prefix+choice["text"]+suffix for choice in gen_obj["choices"]]
        if list_append:
            value_list = variable_stack.get(name, [])
            value_list.append(generated_values)
            if logprobs is not None:
                logprobs_list = variable_stack.get(name+"_logprobs", [])
                logprobs_list.append([choice["logprobs"]["top_logprobs"] for choice in gen_obj["choices"]])
        elif name is not None:
            variable_stack[name] = generated_values
            if logprobs is not None:
                variable_stack[name+"_logprobs"] = [choice["logprobs"]["top_logprobs"] for choice in gen_obj["choices"]]

        if not hidden:
            # TODO: we could enable the parsing to branch into multiple paths here, but for now we just complete the program with the first prefix
            generated_value = generated_values[0]

            # echoing with multiple completions is not standard behavior
            # this just uses the first generated value for completion and the rest as alternatives only used for the variable storage
            # we mostly support this so that the echo=False hiding behavior does not make multiple outputs more complicated than it needs to be in the UX
            # if echo:
            #     variable_stack["@raw_prefix"] += generated_value
            
            id = uuid.uuid4().hex
            l = len(generated_values)
            out = "{{!--" + f"GMARKERmany_generate_start_{not hidden}_{l}${id}$" + "--}}"
            for i, value in enumerate(generated_values):
                if i > 1:
                    out += "--}}"
                if i > 0:
                    out += "{{!--" + f"GMARKERmany_generate_{not hidden}_{i}${id}$" + "--}}{{!--G "
                    out += escape_template_block(value)
                else:
                    out += value
            variable_stack["@raw_prefix"] += out + "--}}{{!--" + f"GMARKERmany_generate_end${id}$" + "--}}"
            return
            # return "{{!--GMARKERmany_generate_start$$}}" + "{{!--GMARKERmany_generate$$}}".join([v for v in generated_values]) + "{{!--GMARKERmany_generate_end$$}}"
            # return "".join([v for v in generated_values])
        else:
            # pop off the variable context we pushed since we are hidden
            variable_stack.pop()