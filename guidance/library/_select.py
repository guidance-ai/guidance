import itertools
import pygtrie
import numpy as np

async def select(variable_name="selected", options=None, logprobs=None, list_append=False, _parser_context=None):
    ''' Select a value from a list of choices.

    Parameters
    ----------
    variable_name : str
        The name of the variable to set with the selected value.
    options : list of str or None
        An optional list of options to select from. This argument is only used when select is used in non-block mode.
    logprobs : str or None
        An optional variable name to set with the logprobs for each option. If this is set the log probs of every option
        is fully evaluated. When this is None (the default) we use a greedy max approach to select the option (similar to
        how greedy decoding works in a language model). So in some cases the selected option can change when logprobs is
        set since it will be more like an exhaustive beam search scoring than a greedy max scoring.
    list_append : bool
        Whether to append the generated value to a list stored in the variable. If set to True, the variable
        must be a list, and the generated value will be appended to the list.
    '''
    parser = _parser_context['parser']
    block_content = _parser_context['block_content']
    parser_prefix = _parser_context['parser_prefix']
    partial_output = _parser_context['partial_output']
    next_node = _parser_context["next_node"]
    next_next_node = _parser_context["next_next_node"]

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

    # find what text follows the select command and append it to the options.
    # we do this so we can differentiate between select options where one is a prefix of another
    next_text = next_node.text if next_node is not None else ""
    if next_next_node and next_next_node.text.startswith("{{~"):
        next_text = next_text.lstrip()
        if next_next_node and next_text == "":
            next_text = next_next_node.text
    if next_text == "": # if we have nothing after us then we are at the end of the text
        next_text = parser.program.llm.end_of_text()
    options = [option + next_text for option in options]

    # TODO: this retokenizes the whole prefix many times, perhaps this could become a bottleneck?
    options_tokens = [parser.program.llm.encode(parser_prefix + option, fragment=False) for option in options]

    # encoding the prefix and then decoding it might change the length, so we need to account for that
    recoded_parser_prefix_length = len(parser.program.llm.decode(parser.program.llm.encode(parser_prefix, fragment=False), fragment=False))

    # build a trie of the options
    token_map = pygtrie.Trie()
    for i,option in enumerate(options_tokens):
        token_map[option] = i
    
    async def recursive_select(current_prefix, allow_token_extension=True):
        """ This returns a dictionary of scores for each option (keyed by the option index).
        """

        # find which select options are possible
        try:
            extension_options = token_map.items(prefix=current_prefix)
        except KeyError:
            return {}

        # this is the dictionary of logprobs for each option we will return
        # note that the logprobs are just for this branch point and below in the decision tree
        logprobs_out = {option[0]: -1000 for option in extension_options}

        # extend the prefix with the longest common prefix among the valid options
        # we also stop early if we have one option
        if len(extension_options) == 1:
            logprobs_out[extension_options[0][0]] = 0 # probability of 1.0 that we will select the only valid option
            return logprobs_out
        else:
            match_index = len(current_prefix)
            for i in range(len(current_prefix), min([len(o[0]) for o in extension_options])):
                if len(set([o[0][i] for o in extension_options])) > 1:
                    break
                match_index += 1
            if match_index > len(current_prefix):
                current_prefix += extension_options[0][0][len(current_prefix):match_index]
                # extension_options = [(option[i:], index) for option,index in extension_options]

        # bias the logits towards valid options
        logit_bias = {}
        for option_tokens,index in extension_options:
            logit_bias[option_tokens[match_index]] = 100

        # check for where we are at the end of the prefix
        if len(logit_bias) == 0 and current_prefix in [o[0] for o in extension_options]:
            logprobs_out[current_prefix] = 0
            return logprobs_out

        # generate the token logprobs
        gen_obj = await parser.llm_session(
            parser.program.llm.decode(current_prefix, fragment=False), # TODO: perhaps we should allow passing of token ids directly? (this could allow us to avoid retokenizing the whole prefix many times)
            max_tokens=1,
            logit_bias=logit_bias,
            logprobs=len(logit_bias),
            cache_seed=0,
            token_healing=False # we manage token boundary healing ourselves for this function
        )
        gen_obj = gen_obj["choices"][0] # get the first choice (we only asked for one)
        if "logprobs" in gen_obj:
            logprobs_result = gen_obj["logprobs"]
            
            # convert the logprobs keys from string back to token ids
            top_logprobs = {}
            for k,v in logprobs_result["top_logprobs"][0].items():
                id = parser.program.llm.token_to_id(k)
                top_logprobs[id] = v
        
        # this happens if LLM does not return logprobs (like an OpenAI chat model)
        else:
            assert logprobs is None, "You cannot ask for the logprobs in a select call when using a model that does not return logprobs!"
            top_logprobs = {parser.program.llm.token_to_id(gen_obj["text"]): 0}
        
        # no need to explore all branches if we are just taking the greedy max
        if logprobs is None:
            max_key = max(top_logprobs, key=top_logprobs.get)
            top_logprobs = {max_key: top_logprobs[max_key]}

        # for each possible next token, see if it grows the prefix in a valid way
        for token,logprob in top_logprobs.items():
            sub_logprobs = await recursive_select(current_prefix + [token])

            # we add the logprob of this token to the logprob of the suffix
            for k in sub_logprobs:

                # compute the probability of a logical OR between the new extension and the previous possible ones
                p1 = np.exp(logprobs_out[k])
                p2 = np.exp(sub_logprobs[k] + logprob)
                or_prob = p1 + p2 - p1*p2
                logprobs_out[k] = np.log(or_prob)

        return logprobs_out
        
    # recursively compute the logprobs for each option
    option_logprobs = await recursive_select([])

    # convert the key from a token list to a string
    option_logprobs = {parser.program.llm.decode(k, fragment=False): v for k,v in option_logprobs.items()}

    # trim off the prefix and suffix we added to the options
    option_logprobs = {k[recoded_parser_prefix_length:len(k)-len(next_text)]: v for k,v in option_logprobs.items()}

    # select the option with the highest logprob
    selected_option = max(option_logprobs, key=option_logprobs.get)
    
    # see if we are appending to a list or not
    if list_append:
        value_list = parser.get_variable(variable_name, [])
        value_list.append(selected_option)
        parser.set_variable(variable_name, value_list)
        if logprobs is not None:
            logprobs_list = parser.get_variable(logprobs, [])
            logprobs_list.append(option_logprobs)
            parser.set_variable(logprobs, logprobs_list)
    else:
        parser.set_variable(variable_name, selected_option)
        if logprobs is not None:
            parser.set_variable(logprobs, option_logprobs)
    
    if max(option_logprobs.values()) <= -1000:
        raise ValueError("No valid option generated in #select! Please post a GitHub issue since this should not happen :)")
    
    partial_output(selected_option)

select.is_block = True