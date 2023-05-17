import itertools

async def select(variable_name="selected", options=None, logprobs=None, _parser_context=None):
    ''' Select a value from a list of choices.

    Parameters
    ----------
    variable_name : str
        The name of the variable to set with the selected value.
    options : list of str or None
        An optional list of options to select from. This argument is only used when select is used in non-block mode.
    logprobs : str or None
        An optional variable name to set with the logprobs for each option.
    '''
    parser = _parser_context['parser']
    block_content = _parser_context['block_content']
    parser_prefix = _parser_context['parser_prefix']
    partial_output = _parser_context['partial_output']

    if block_content is None:
        assert options is not None, "You must provide an options list like: {{select 'variable_name' options}} when using the select command in non-block mode."
    else:
        assert len(block_content) > 1, "You must provide at least one two options to the select block command."
        assert options is None, "You cannot provide an options list when using the select command in block mode."

    if options is None:
        options = [block_content[0].text]
        for i in range(1, len(block_content), 2):
            assert block_content[i].text == "{{or}}"
            options.append(block_content[i+1].text)

    option_tokens = [parser.program.llm.encode(option) for option in options]
    ids_used = set(itertools.chain.from_iterable(option_tokens))

    # find the common prefix of all the options BROKEN STILLL
    max_tokens = max([len(o) for o in option_tokens])
    # for i in range(max_tokens):
    #     all_match = True
    #     pos_val = None
    #     for j in range(len(option_tokens)):
    #         if len(option_tokens[j]) <= i:
    #             if pos_val is None:
    #                 pos_val = option_tokens[j][i]
    #             elif option_tokens[j][i] != pos_val:
    #                 all_match = False
    #                 break
    #     if not all_match:
    #         max_tokens = i
    #         break



    # call the session to get the logprobs for each option
    gen_obj = await parser.llm_session(
        parser_prefix,
        max_tokens=max([len(o) for o in option_tokens]),
        logit_bias={str(id): 50 for id in ids_used},
        logprobs=10,
        cache_seed=0
    )
    logprobs_result = gen_obj["choices"][0]["logprobs"]
    top_logprobs = logprobs_result["top_logprobs"]

    # add the token healing prefix if it exists
    remove_prefix = 0
    if gen_obj is not None and "token_healing_prefix" in logprobs_result:
        remove_prefix = len(logprobs_result["token_healing_prefix"])
        options = [logprobs_result["token_healing_prefix"] + option for option in options]
        option_tokens = [parser.program.llm.encode(option) for option in options]
    
    # initialize the option logprobs
    option_logprobs = {}
    for option in option_tokens:
        option_logprobs[parser.program.llm.decode(option)] = 0

    # compute logprobs for each option
    for i in range(len(top_logprobs)):
        for option_ids,option_str in zip(option_tokens, options):
            if len(option_ids) > i:
                # if i == 0 and "token_healing_prefix" in logprobs_result:
                #     logprobs_result
                # option_string = parser.program.llm.decode(option_ids)
                option_logprobs[option_str] += top_logprobs[i].get(parser.program.llm.decode([option_ids[i]]), -1000)
    
    # penalize options that are too long
    for option in option_tokens:
        if len(option) > len(top_logprobs):
            option_logprobs[parser.program.llm.decode(option)] -= 1000

    selected_option = max(option_logprobs, key=option_logprobs.get)[remove_prefix:]
    parser.set_variable(variable_name, selected_option)
    if logprobs is not None:
        parser.set_variable(logprobs, option_logprobs)
    
    if max(option_logprobs.values()) <= -1000:
        raise ValueError("No valid option generated in #select, this could be fixed if we used a tokenizer and forced the LM to use a valid option! The top logprobs were" + str(top_logprobs))
    
    partial_output(selected_option)

    return selected_option
select.is_block = True