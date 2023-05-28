import inspect
import re
import uuid
from .._utils import strip_markers

async def geneach(list_name, stop=None, max_iterations=100, min_iterations=0, num_iterations=None, hidden=False, join="", single_call=False, single_call_temperature=0.0, single_call_max_tokens=500, single_call_top_p=1.0, _parser_context=None):
    ''' Generate a potentially variable length list of items using the LLM.

    Parameters
    ----------
    list_name : str
        The name of the variable to save the generated list to.
    stop : str or list of str
        A string or list of strings that will stop the generation of the list. For example if stop="</ul>"
        then the list will be generated until the first "</ul>" is generated.
    max_iterations : int
        The maximum number of items to generate.
    min_iterations : int
        The minimum number of items to generate.
    num_iterations : int
        The exact number of items to generate (this overrides max_iterations and min_iterations).
    hidden : bool
        If True, the generated list items will not be added to the LLMs input context. This means that each
        item will be generated independently of the others. Note that if you use hidden=True you must also
        set num_iterations to a fixed number (since without adding items the context there is not way for the
        LLM to know when to stop on its own).
    join : str
        A string to join the generated items with.
    single_call : bool
        This is an option designed to make look generation more convienent for LLMs that don't support guidance
        acceleration. If True, the LLM will be called once to generate the entire list. This only works if the
        LLM has already been prompted to generate content that matches the format of the list. After the single
        call, the generated list variables will be parsed out of the generated text using a regex. (note that only
        basic template tags are supported in the list items when using single_call=True).
    single_call_temperature : float
        Only used with single_call=True. The temperature to use when generating the list items in a single call.
    single_call_max_tokens : int
        Only used with single_call=True. The maximum number of tokens to generate when generating the list items.
    single_call_top_p : float
        Only used with single_call=True. The top_p to use when generating the list items in a single call.
    
    '''
    block_content = _parser_context["block_content"]
    parser = _parser_context["parser"]
    partial_output = _parser_context["partial_output"]
    parser_prefix = _parser_context["parser_prefix"]
    parser_node = _parser_context["parser_node"]

    assert len(block_content) == 1
    assert not (hidden and single_call), "Cannot use hidden=True and single_call together"
    assert isinstance(list_name, str), "Must provide a variable name to save the generated list to"
    assert not hidden or num_iterations is not None, "Cannot use hidden=True and variable length iteration together yet..."
    echo = not hidden

    # num_iterations has priority over max_iterations if they are both set
    if num_iterations is not None:
        max_iterations = num_iterations
        min_iterations = num_iterations

    # give the list a default name
    if list_name is None:
        list_name = 'generated_list'

    # if stop is None then we use the text of the node after the generate command
    # if stop is None:

    #     next_text = next_node.text if next_node is not None else ""
    #     prev_text = prev_node.text if prev_node is not None else ""

    #     # auto-detect quote stop tokens
    #     quote_types = ['"', "'", "'''", '"""', "`"]
    #     for quote_type in quote_types:
    #         if next_text.startswith(quote_type) and prev_text.endswith(quote_type):
    #             stop = quote_type
    #             break
                
    #     # auto-detect XML tag stop tokens
    #     if stop is None:
    #         m = re.match(r"^\s*(</[^>]+>)", next_text, re.DOTALL) #next_text.startswith(end_tag)
    #         if m is not None:
    #             stop = m.group(1)
            
    #         m = re.match(r"^\s*(<|im_end|>)", next_text, re.DOTALL) #next_text.startswith(end_tag)
    #         if m is not None:
    #             stop = "<|im_end|>"
            
    #         if next_text != "":
    #             stop = next_text

    out = []
    partial_out = ""
    
    # convert stop strings to tokens
    if stop is not False:
        if stop is None:
            max_stop_tokens = 2
        else:
            max_stop_tokens = max([len(parser.program.llm.encode(s)) for s in stop]) + 2

    if not single_call:
        i = 0
        data = []
        while True:
            parser.variable_stack.append({})
            parser.variable_stack[-1]["@index"] = i
            parser.variable_stack[-1]["@first"] = i == 0
            parser.variable_stack[-1]["this"] = {}
            pos = len(parser.prefix)
            
            # add the join string if we are not on the first iteration
            if i > 0 and join != "":
                partial_output(join)
            
            await parser.visit(block_content[0]) # fills out parser.prefix
            block_variables = parser.variable_stack.pop()["this"]

            # update the list variable (we do this each time we get a new item so that streaming works)
            parser.set_variable(list_name, parser.get_variable(list_name, default_value=[]) + [block_variables])
            
            if hidden:
                # new_content = parser.prefix[pos:]
                parser.reset_prefix(pos)
            if not parser.executing:
                block_text = parser.prefix[pos:]
                block_text = block_text # make any unfinished this. references point to the last (unfinished) item
                parser.prefix = parser.prefix[:pos] + parser.prefix[pos:].replace("this.", list_name+"[-1].")
                break
            i += 1

            # see if we hit an await and so are not executing anymore
            if not parser.executing:
                break

            # check if the block has thrown a stop iteration signal
            if parser.caught_stop_iteration:
                parser.caught_stop_iteration = False
                break

            # check if we have reached the max iterations
            if max_iterations is not None and i >= max_iterations:
                break

            # we run a quick generation to see if we have reached the end of the list (note the +2 tokens is to help be tolorant to whitespace)
            if stop is not False and i >= min_iterations:
                try:
                    gen_obj = await parser.llm_session(strip_markers(parser.prefix), stop=stop, max_tokens=max_stop_tokens, temperature=0, cache_seed=0)
                except Exception:
                    raise Exception(f"Error generating stop tokens for geneach loop. Perhaps you are outside of role tags (assistant/user/system)? If you don't want the loop to check for stop tokens, set stop=False or set num_iterations.")
                if gen_obj["choices"][0]["finish_reason"] == "stop":
                    break
    
    # TODO: right now single_call is a bit hacky, we should make it more robust to rich loop item template structures
    else: # if single_call
        # create a pattern to match each item
        pattern = re.sub(
            r'{{gen [\'"]([^\'"]+)[\'"][^}]*}}',
            lambda x: r"(?P<"+_escape_group_name(x.group(1))+">.*?)",
            block_content[0].text
        )

        # fixed prefixes can be used if we know we have at least one iteration
        if min_iterations > 0:
            # find what part of the pattern is fixed before the first generation
            fixed_prefix = re.match(r"^(.*)\(\?P\<", pattern, flags=re.DOTALL)[0][:-4]
            fixed_prefix = fixed_prefix.replace(r"{{@index}}", "0") # TODO: this is a bit hacky
        else:
            fixed_prefix = ""

        # assume the LLM will also generate whatever interpolations are in the pattern
        pattern = re.sub(r"{{(.*?)}}", lambda x: r"(?P<" + _escape_group_name(x.group(1)) + ">.*?)", pattern)

        # generate the looped content
        if single_call_temperature > 0:
            cache_seed = parser.program.cache_seed
            parser.program.cache_seed += 1
        else:
            cache_seed = 0
        gen_stream = await parser.llm_session(parser_prefix+fixed_prefix, stop=stop, max_tokens=single_call_max_tokens, temperature=single_call_temperature, top_p=single_call_top_p, cache_seed=cache_seed, stream=True)
        generated_value = fixed_prefix
        num_items = 0
        data = []
        for gen_obj in gen_stream:
            generated_value += gen_obj["choices"][0]["text"]


            # parse the generated content (this assumes the generated content is syntactically correct)
            matches = re.finditer(pattern, generated_value)
            for m in matches:#"{{!--" + f"GMARKER_START_{name}${node_text}$}}{out}{{!--GMARKER_END_{name}$$" + "}}"
                
                # consume the generated value up to the match
                generated_value = generated_value[m.end():]

                # get the variables that were generated
                match_dict = m.groupdict()
                if "this" in match_dict:
                    next_item = match_dict["this"]
                else:
                    d = {}
                    for k in match_dict:
                        k = _unescape_group_name(k)
                        if k.startswith("this."):
                            d[k[5:]] = match_dict[k].strip()
                    next_item = d

                # update the list variable (we do this each time we get a new item so that streaming works)
                parser.set_variable(list_name, parser.get_variable(list_name, default_value=[]) + [next_item])

                # recreate the output string with format markers added
                item_out = re.sub(
                    r"{{(?!~?gen)(.*?)}}",
                    lambda x: match_dict[_escape_group_name(x.group(1))],
                    block_content[0].text
                )
                item_out = re.sub(
                    r"{{gen [\'\"]([^\'\"]+)[\'\"][^}]*}}",
                    lambda x: "{{!--GMARKER_START_gen$"+x.group().replace("$", "&#36;").replace("{", "&#123;").replace("}", "&#125;")+"$--}}"+match_dict[_escape_group_name(x.group(1))]+"{{!--GMARKER_END_gen$$--}}",
                    item_out
                )
                partial_output("{{!--GMARKER_each$$--}}" + item_out) # marker and content of the item
                num_items += 1
                # out.append(item_out)

                # if we have hit the max iterations, stop the LLM
                if num_items >= max_iterations:
                    gen_stream.close()
    
    partial_output("{{!--GMARKER_each$$--}}") # end marker

    # parser.get_variable(list, [])
    #parser.set_variable(list_name, parser.get_variable(list_name, default_value=[]) + data)
   
    # if we have stopped executing, we need to add the loop to the output so it can be executed later
    if not parser.executing:
        if isinstance(list_name, str):
            partial_output(parser_node.text)

    return ""
    
    if echo:
        return "{{!--GMARKER_each$$--}}" + "{{!--GMARKER_each$$--}}".join(out) + "{{!--GMARKER_each$$--}}" + suffix
    else:
        id = uuid.uuid4().hex
        l = len(out)
        out_str = prefix + "{{!--" + f"GMARKER_each_noecho_start_{echo}_{l}${id}$" + "--}}"
        for i, value in enumerate(out):
            if i > 0:
                out_str += "{{!--" + f"GMARKER_each_noecho_{echo}_{i}${id}$" + "--}}"
            out_str += value
        return out_str + "{{!--" + f"GMARKER_each_noecho_end${id}$" + "--}}"

        # return "{{!--GMARKER_each_noecho$$}}" + "{{!--GMARKER_each_noecho$$}}".join(out) + "{{!--GMARKER_each_noecho$$}}"
geneach.is_block = True

def _escape_group_name(name):
    return name.replace("@", "_AT_").replace(".", "_DOT_")

def _unescape_group_name(name):
    return name.replace("_AT_", "@").replace("_DOT_", ".")