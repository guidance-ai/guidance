import inspect
import re
import uuid
import asyncio
from .._utils import ContentCapture

async def each(list, hidden=False, parallel=False, filter=None, _parser_context=None):
    ''' Iterate over a list and execute a block for each item.

    Parameters
    ----------
    list : iterable
        The list to iterate over. Inside the block each element will be available as `this`.
    hidden : bool
        Whether to include the generated item blocks in future LLM context.
    parallel : bool
        If this is True then we generate all the items in the list in parallel. Note that this
        is only compatible with hidden=True. When parallel=True you can no longer raise a
        StopIteration exception to stop the loop at a specific step (since the steps can be run
        in parallel in any order).
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
    variable_stack = _parser_context['variable_stack']
    assert len(block_content) == 1

    assert not parallel or hidden is True, "parallel=True is only compatible with hidden=True (since if hidden=False earlier items are contex for later items)"

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

    # we parse in parallel if the loop contents are hidden
    if parallel:
        
         # set up the coroutines to call
        coroutines = []
        for i, item in enumerate(list):
            variable_stack.push({
                "@index": i,
                "@first": i == 0,
                "@last": i == len(list) - 1,
                "this": item
            })
            coroutines.append(parser.visit(
                block_content[0],
                variable_stack,
                next_node=_parser_context["next_node"],
                next_next_node=_parser_context["next_next_node"],
                prev_node=_parser_context["prev_node"],
                hidden=True
            ))
            variable_stack.pop()

        item_outs = await asyncio.gather(*coroutines)
        
        for item_out in item_outs:

            # parser._trim_prefix(item_out)
            out.append(item_out)

            # check if the block has thrown a stop iteration signal
            if parser.caught_stop_iteration:
                parser.caught_stop_iteration = False
                break
    else:
        for i, item in enumerate(list):

            variable_stack.push({
                "@index": i,
                "@first": i == 0,
                "@last": i == len(list) - 1,
                "this": item
            })
            with ContentCapture(variable_stack, hidden) as new_content:
                new_content += await parser.visit(
                    block_content[0],
                    variable_stack,
                    next_node=_parser_context["next_node"],
                    next_next_node=_parser_context["next_next_node"],
                    prev_node=_parser_context["prev_node"]
                )
                out.append(str(new_content))

            # check if the block has thrown a stop iteration signal
            if parser.caught_stop_iteration:
                parser.caught_stop_iteration = False
                break
            variable_stack.pop()
    
    # suffix = ""
    # if not parser.executing:
    #     if isinstance(list, str):
    #         variable_stack["_prefix"] += parser_node.text # TODO: we should update the params of the loop iteration here
    #         # suffix = partial_out + parser_node.text
    
    # if not hidden:
        #return "{{!--GMARKER_each$$--}}" + "{{!--GMARKER_each$$--}}".join(out) + "{{!--GMARKER_each$$--}}" + suffix
    if hidden:
        id = uuid.uuid4().hex
        l = len(out)
        out_str = "{{!--" + f"GMARKER_each_noecho_start_{not hidden}_{l}${id}$" + "--}}"
        for i, value in enumerate(out):
            if i > 0:
                out_str += "{{!--" + f"GMARKER_each_noecho_{not hidden}_{i}${id}$" + "--}}"
            out_str += value
        return out_str + "{{!--" + f"GMARKER_each_noecho_end${id}$" + "--}}"

        # return "{{!--GMARKER_each_noecho$$}}" + "{{!--GMARKER_each_noecho$$}}".join(out) + "{{!--GMARKER_each_noecho$$}}"
each.is_block = True