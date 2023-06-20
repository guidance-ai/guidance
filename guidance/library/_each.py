import asyncio
import builtins
from .._utils import ContentCapture

async def each(list, hidden=False, parallel=False, item_name="this", start_index=0, _parser_context=None):
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
    item_name : str
        The name of the variable to use for the current item in the list.
    '''
    block_content = _parser_context['block_content']
    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']

    assert not parallel or hidden is True, "parallel=True is only compatible with hidden=True (since if hidden=False earlier items are contex for later items)"

    if isinstance(list, dict):
        keys = builtins.list(list.keys())
        list = list.values()
    else:
        keys = None

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
            if i < start_index: # skip items before the start index
                continue
            context = {
                "@index": i,
                "@first": i == 0,
                "@last": i == len(list) - 1,
                item_name: item,
                "@raw_prefix": variable_stack["@raw_prefix"], # create a local copy of the prefix since we are hidden
                "@no_display": True
            }
            if keys is not None:
                context["@key"] = keys[i]
            variable_stack.push(context)
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
            context = {
                "@index": i,
                "@first": i == 0,
                "@last": i == len(list) - 1,
                item_name: item
            }
            if keys is not None:
                context["@key"] = keys[i]
            variable_stack.push(context)
            with ContentCapture(variable_stack, hidden) as new_content:
                new_content += await parser.visit(
                    block_content,
                    variable_stack,
                    next_node=_parser_context["next_node"],
                    next_next_node=_parser_context["next_next_node"],
                    prev_node=_parser_context["prev_node"]
                )
                out.append(str(new_content))
            variable_stack.pop()

            # if we stopped executing then we need to dump our node text back out but with the start_index incremented to account for what we've already done
            if not parser.executing:
                updated_text = re.sub(r"^({{~?#each.*?)(~?}})", r"\1 start_index="+str(i+1)+r"\2", _parser_context["parser_node"].text)
                variable_stack["@raw_prefix"] += updated_text
                break

            # check if the block has thrown a stop iteration signal
            if parser.caught_stop_iteration:
                parser.caught_stop_iteration = False
                break
        
            
    
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