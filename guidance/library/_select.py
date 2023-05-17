import itertools
import pygtrie
import numpy as np

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

    # build a trie of the options
    token_map = pygtrie.CharTrie()
    for i,option in enumerate(options):
        token_map[option] = i
    
    async def rec_select(current_prefix):
        """ This returns a dictionary of scores for each option (keyed by the option index).
        """

        # this is the dictionary of logprobs for each option we will return
        # note that the logprobs are just for this branch point and below in the decision tree
        # logprobs_out = {option: -100 for option in options}

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
            for i in range(len(current_prefix), min([len(o[0]) for o in extension_options])):
                if len(set([o[0][i] for o in extension_options])) > 1:
                    break
            current_prefix = extension_options[0][0][:i]

        # extend the prefix by one token using the model
        gen_obj = await parser.llm_session(
            parser_prefix + current_prefix,
            max_tokens=1,
            # logit_bias={str(id): 50 for id in ids_used},
            logprobs=10,
            cache_seed=0
        )
        logprobs_result = gen_obj["choices"][0]["logprobs"]
        top_logprobs = logprobs_result["top_logprobs"][0]

        remove_prefix = len(logprobs_result.get("token_healing_prefix", ""))

        # for each possible next token, see if it grows the prefix in a valid way
        for token_str,logprob in top_logprobs.items():
            if len(token_str[remove_prefix:]) > 0:
                sub_logprobs = await rec_select(current_prefix + token_str[remove_prefix:])
                for k in sub_logprobs:
                    logprobs_out[k] = sub_logprobs[k] + logprob

        # if we did token healing and did not extend past our prefix we need to consider the next token
        # TODO: when returning all logprobs we need to consider all the options, which means we should
        # force the model to not token heal and see what would have happened then on the next token...
        first_token_str = max(top_logprobs, key=top_logprobs.get)
        if len(logprobs_result["top_logprobs"]) > 1 and len(first_token_str) == remove_prefix:            
            top_logprobs = logprobs_result["top_logprobs"][1]
            for token_str,logprob in top_logprobs.items():
                sub_logprobs = await rec_select(current_prefix + token_str)
                for k in sub_logprobs:

                    # compute the probability of a logical OR between the new extension and the previous possible ones
                    p1 = np.exp(logprobs_out[k])
                    p2 = np.exp(sub_logprobs[k] + logprob)
                    or_prob = p1 + p2 - p1*p2
                    logprobs_out[k] = np.log(or_prob)

        return logprobs_out
        
    # recursively compute the logprobs for each option
    option_logprobs = await rec_select("") 

    selected_option = max(option_logprobs, key=option_logprobs.get)
    parser.set_variable(variable_name, selected_option)
    if logprobs is not None:
        parser.set_variable(logprobs, option_logprobs)
    
    if max(option_logprobs.values()) <= -1000:
        raise ValueError("No valid option generated in #select, this could be fixed if we used a tokenizer and forced the LM to use a valid option! The top logprobs were" + str(top_logprobs))
    
    partial_output(selected_option)

select.is_block = True