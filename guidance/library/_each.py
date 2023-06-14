import inspect
import re
import uuid
import asyncio
from .._utils import ContentCapture

async def each(list, hidden=False, parallel=False, _parser_context=None):
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
    '''
    block_content = _parser_context['block_content']
    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']

    assert not parallel or hidden is True, "parallel=True is only compatible with hidden=True (since if hidden=False earlier items are contex for later items)"

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
                "this": item,
                "@raw_prefix": variable_stack["@raw_prefix"], # create a local copy of the prefix since we are hidden
                "_no_display": True
            })
            coroutines.append(parser.visit(
                block_content,
                variable_stack.copy(),
                next_node=_parser_context["next_node"],
                next_next_node=_parser_context["next_next_node"],
                prev_node=_parser_context["prev_node"]
            ))
            variable_stack.pop()

        await asyncio.gather(*coroutines)
        
            # for item_out in item_outs:

            #     # parser._trim_prefix(item_out)
            #     out.append(item_out)

            #     # check if the block has thrown a stop iteration signal
            #     if parser.caught_stop_iteration:
            #         parser.caught_stop_iteration = False
            #         break
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
                    block_content,
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
    
    # if not hidden:
        #return "{{!--GMARKER_each$$--}}" + "{{!--GMARKER_each$$--}}".join(out) + "{{!--GMARKER_each$$--}}" + suffix
    # if hidden:
    #     id = uuid.uuid4().hex
    #     l = len(out)
    #     out_str = "{{!--" + f"GMARKER_each_noecho_start_{not hidden}_{l}${id}$" + "--}}"
    #     for i, value in enumerate(out):
    #         if i > 0:
    #             out_str += "{{!--" + f"GMARKER_each_noecho_{not hidden}_{i}${id}$" + "--}}"
    #         out_str += value
    #     return out_str + "{{!--" + f"GMARKER_each_noecho_end${id}$" + "--}}"

each.is_block = True