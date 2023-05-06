import inspect
import re
import uuid
from .._utils import strip_markers

async def geneach(list_name, block_content, parser, partial_output=None, parser_prefix=None, parser_node=None, stop=None, max_iterations=100, min_iterations=0, num_iterations=None, hidden=False, filter=None, join="", single_call=False, single_call_temperature=0.0, single_call_max_tokens=500, single_call_top_p=1.0, next_node=None, prev_node=None):
    ''' Generate a list of items.
    '''
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
            stop_tokens = []
        else:
            stop_tokens = [parser.program.llm.encode(s) for s in stop]

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
            if len(data) > 0 and join != "":
                partial_output(join)
            
            await parser.visit(block_content[0]) # fills out parser.prefix
            block_variables = parser.variable_stack.pop()["this"]
            data.append(block_variables)
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
                gen_obj = await parser.llm_session(strip_markers(parser.prefix), stop=stop, max_tokens=len(stop_tokens)+2, temperature=0, cache_seed=0)
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
                    data.append(match_dict["this"])
                else:
                    d = {}
                    for k in match_dict:
                        k = _unescape_group_name(k)
                        if k.startswith("this."):
                            d[k[5:]] = match_dict[k].strip()
                    data.append(d)

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
    parser.set_variable(list_name, parser.get_variable(list_name, default_value=[]) + data)
   
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