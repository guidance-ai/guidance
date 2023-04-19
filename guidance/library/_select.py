def select(variable_name="selected", block_content=None, parser=None, partial_output=None, parser_prefix=None, logprobs=None):
    ''' Select a value from a list of choices.
    '''
    assert len(block_content) > 1
    options = [block_content[0].text]
    for i in range(1, len(block_content), 2):
        assert block_content[i].text == "{{or}}"
        options.append(block_content[i+1].text)

    option_tokens = [parser.program.llm.encode(option) for option in options]

    # [TODO] we should force the LM to generate a valid specific option
    #        for openai this means setting logprobs to valid token ids
    gen_obj = parser.llm_session(
        parser_prefix,
        max_tokens=max([len(o) for o in option_tokens]),
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
        option_tokens = [parser.program.llm.encode(option, is_suffix=True) for option in options]
    
    # initialize the option logprobs
    option_logprobs = {}
    for option in option_tokens:
        option_logprobs[parser.program.llm.decode(option, is_suffix=True)] = 0

    # compute logprobs for each option
    for i in range(len(top_logprobs)):
        for option_ids,option_str in zip(option_tokens, options):
            if len(option_ids) > i:
                # if i == 0 and "token_healing_prefix" in logprobs_result:
                #     logprobs_result
                # option_string = parser.program.llm.decode(option_ids)
                option_logprobs[option_str] += top_logprobs[i].get(parser.program.llm.decode([option_ids[i]], is_suffix=True), -1000)
    
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