import inspect
import re
import uuid
from .._utils import strip_markers

async def each(list, block_content, parser, parser_prefix=None, parser_node=None, stop=None, hidden=False, filter=None, batch_generate=False, batch_generate_temperature=0.0, batch_generate_max_tokens=500, batch_generate_top_p=1.0):
    ''' Iterate over a list and execute a block for each item.
    '''
    assert len(block_content) == 1
    assert not (hidden and batch_generate), "Cannot use hidden=True and batch_generate together"
    assert not (hidden and isinstance(list, str)), "Cannot use hidden=True and variable length iteration together" # TODO: perhaps we can relax this?
    echo = not hidden

    # an empty string means the variable was not provided
    if list is None:
        list = []

    # apply a filter to the list if one was provided
    if filter is not None:

        # if the list is callable then we call it to get an iterable
        if callable(filter):
            # we support a set of optional arguments to the list function
            named_args = {}
            sig = inspect.signature(filter)
            if "template_context" in sig.parameters:
                named_args["template_context"] = {
                    "@block_text": block_content[0].text,
                    "@tokenizer": parser.program.tokenizer
                }
            list = filter(list, **named_args)
        else:
            raise TypeError("Can't apply a non-callable filter: " + str(filter))

    # make sure the list is iterable
    try:
        iter(list)
    except TypeError:
        raise TypeError("The #each command cannot iterate over a non-iterable value: " + str(list))

    out = []
    partial_out = ""
    
    # if the list is a string then it is the name of a variable to save a new list to
    if isinstance(list, str):
        assert False, 'Use #geneach instead of #each for variable length iteration'
        # if stop is None:
        #     stop = "<|endoftext|>"
        # assert stop is not None, "Must provide a stop token when doing variable length iteration!"
        if stop is not None:
            stop_tokens = [parser.program.llm.encode(s) for s in stop]

        if not batch_generate:
            i = 0
            data = []
            while True:
                parser.variable_stack.append({})
                parser.variable_stack[-1]["@index"] = i
                parser.variable_stack[-1]["@first"] = i == 0
                parser.variable_stack[-1]["this"] = {}
                pos = len(parser.prefix)
                await parser.visit(block_content[0]) # fills out parser.prefix
                block_variables = parser.variable_stack.pop()["this"]
                data.append(block_variables)
                if not parser.executing:
                    block_text = parser.prefix[pos:]
                    block_text = block_text # make any unfinished this. references point to the last (unfinished) item
                    parser.prefix = parser.prefix[:pos] + parser.prefix[pos:].replace("this.", list+"[-1].")
                    break
                i += 1

                # see if we hit an await and so are not executing anymore
                if not parser.executing:
                    break

                # check if the block has thrown a stop iteration signal
                if parser.caught_stop_iteration:
                    parser.caught_stop_iteration = False
                    break

                # we run a quick generation to see if we have reached the end of the list (note the +2 tokens is to help be tolorant to whitespace)
                if stop is not None:
                    gen_obj = parser.program.llm(strip_markers(parser.prefix), stop=stop, max_tokens=len(stop_tokens)+2, temperature=0, cache_seed=0)
                    if gen_obj["choices"][0]["finish_reason"] == "stop":
                        break
        else:
            # create a pattern to match each item
            pattern = re.sub(
                r'{{gen [\'"]([^\'"]+)[\'"][^}]*}}',
                lambda x: r"(?P<"+x.group(1).replace("this.", "")+">.*?)",
                block_content[0].text
            )

            # generate the looped content
            if batch_generate_temperature > 0:
                cache_seed = parser.program.cache_seed
                parser.program.cache_seed += 1
            else:
                cache_seed = 0
            gen_obj = parser.program.llm(parser_prefix, stop=stop, max_tokens=batch_generate_max_tokens, temperature=batch_generate_temperature, top_p=batch_generate_top_p, cache_seed=cache_seed)
            generated_value = gen_obj["choices"][0]["text"]

            # parse the generated content (this assumes the generated content is syntactically correct)
            matches = re.finditer(pattern, generated_value)
            data = []
            for m in matches:#"{{!--" + f"GMARKER_START_{name}${node_text}$}}{out}{{!--GMARKER_END_{name}$$" + "}}"
                
                # get the variables that were generated
                match_dict = m.groupdict()
                if len(match_dict) == 1 and "this" in match_dict:
                    data.append(match_dict["this"])
                else:
                    data.append(match_dict)

                # recreate the output string with format markers added
                item_out = re.sub(
                    r"{{gen [\'\"]([^\'\"]+)[\'\"][^}]*}}",
                    lambda x: "{{!--GMARKER_START_gen$"+x.group()+"$--}}"+match_dict[x.group(1).replace("this.", "")]+"{{!--GMARKER_END_gen$$--}}",
                    block_content[0].text
                )
                out.append(item_out)
        
        # parser.get_variable(list, [])
        parser.set_variable(list, parser.get_variable(list, default_value=[]) + data)

    # if the list is not a string then it is a list of items to iterate over
    else:
        parser.variable_stack.append({})
        for i, item in enumerate(list):
            parser.variable_stack[-1]["@index"] = i
            parser.variable_stack[-1]["@first"] = i == 0
            parser.variable_stack[-1]["@last"] = i == len(list) - 1
            parser.variable_stack[-1]["this"] = item
            item_out = await parser.visit(block_content[0])
            if not echo:
                parser._trim_prefix(item_out)
            out.append(item_out)
        parser.variable_stack.pop()
    
    suffix = ""
    if not parser.executing:
        if isinstance(list, str):
            suffix = partial_out + parser_node.text
    
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