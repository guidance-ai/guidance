import inspect
import re
import uuid
from .._utils import strip_markers

async def each(list, hidden=False, filter=None, _parser_context=None):
    ''' Iterate over a list and execute a block for each item.

    Parameters
    ----------
    list : iterable
        The list to iterate over. Inside the block each element will be available as `this`.
    hidden : bool
        Whether to include the generated item blocks in future LLM context.
    filter : callable
        A function to filter the list before iterating over it. The function should take a single
        argument (the list) and return a new list. It can also optionally have a `template_context`
        argument that will be set to a dictionary containing the following keys:
            - "@block_text": the text of the block
            - "@tokenizer": the tokenizer used to parse the block
    '''
    block_content = _parser_context['block_content']
    parser = _parser_context['parser']
    parser_node = _parser_context['parser_node']
    assert len(block_content) == 1
    echo = not hidden

    if not parser.executing:
        return parser_node.text

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
    
    parser.variable_stack.append({})
    for i, item in enumerate(list):
        parser.variable_stack[-1]["@index"] = i
        parser.variable_stack[-1]["@first"] = i == 0
        parser.variable_stack[-1]["@last"] = i == len(list) - 1
        parser.variable_stack[-1]["this"] = item
        item_out = await parser.visit(
            block_content[0],
            next_node=_parser_context["next_node"],
            next_next_node=_parser_context["next_next_node"],
            prev_node=_parser_context["prev_node"]
        )
        if not echo:
            parser._trim_prefix(item_out)
        out.append(item_out)

        # check if the block has thrown a stop iteration signal
        if parser.caught_stop_iteration:
            parser.caught_stop_iteration = False
            break
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
        out_str = "{{!--" + f"GMARKER_each_noecho_start_{echo}_{l}${id}$" + "--}}"
        for i, value in enumerate(out):
            if i > 0:
                out_str += "{{!--" + f"GMARKER_each_noecho_{echo}_{i}${id}$" + "--}}"
            out_str += value
        return out_str + "{{!--" + f"GMARKER_each_noecho_end${id}$" + "--}}"

        # return "{{!--GMARKER_each_noecho$$}}" + "{{!--GMARKER_each_noecho$$}}".join(out) + "{{!--GMARKER_each_noecho$$}}"
each.is_block = True