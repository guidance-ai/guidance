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
        An optional variable name to set with the logprobs for each option.
    list_append : bool
        Whether to append the generated value to a list stored in the variable. If set to True, the variable
        must be a list, and the generated value will be appended to the list.
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
            for i in range(len(current_prefix), min([len(o[0]) for o in extension_options])):
                if len(set([o[0][i] for o in extension_options])) > 1:
                    break
            if i > 0:
                current_prefix += extension_options[0][0][:i]
                # extension_options = [(option[i:], index) for option,index in extension_options]

        # bias the logits towards valid options
        tmp_prefix = (parser_prefix[-50:] + current_prefix) # this is so we get the right tokenization at the boundary
        tmp_prefix_tokens = parser.program.llm.encode(tmp_prefix)
        logit_bias1 = {} # token extension biases
        logit_bias2 = {} # next token biases
        for option,index in extension_options:
            option_tokens = parser.program.llm.encode(parser_prefix[-50:] + option)
            
            # if we extended the last token to a longer one
            if option_tokens[len(tmp_prefix_tokens)-1] != tmp_prefix_tokens[-1]:
                if allow_token_extension: # this is a valid extension only if we are not allowed to extend the token
                    logit_bias1[option_tokens[len(tmp_prefix_tokens)-1]] = 100
            
            # if we did not extend the last token to a longer one we can bias the next token
            else:
                if len(option_tokens) > len(tmp_prefix_tokens):
                    logit_bias2[option_tokens[len(tmp_prefix_tokens)]] = 100

            # logit_bias[option_tokens[len(tmp_prefix_tokens)-1]] = tmp_prefix_tokens[-1]
            # if len(option_tokens) > len(tmp_prefix_tokens) and :
            #     logit_bias[option_tokens[len(tmp_prefix_tokens)]] = 100

        # extend the prefix by extending the last token
        if len(logit_bias1) > 0:
            call_prefix = parser.program.llm.decode(tmp_prefix_tokens[:-1])
            logit_bias = logit_bias1
            last_token_str = parser.program.llm.decode(tmp_prefix_tokens[-1:])

        # extend the prefix by adding the next token
        else:
            call_prefix = tmp_prefix
            logit_bias = logit_bias2
            last_token_str = ""

        # generate the token logprobs
        gen_obj = await parser.llm_session(
            call_prefix, # TODO: perhaps we should allow passing of token ids directly?
            max_tokens=1,
            logit_bias=logit_bias,
            logprobs=10,
            cache_seed=0,
            token_healing=False # we manage token boundary healing ourselves for this function
        )
        logprobs_result = gen_obj["choices"][0]["logprobs"]
        top_logprobs = logprobs_result["top_logprobs"][0]

        # for each possible next token, see if it grows the prefix in a valid way
        for token_str,logprob in top_logprobs.items():

            # build our recursive call prefix
            rec_prefix = call_prefix + token_str
            if len(last_token_str) > 0:
                rec_prefix = current_prefix[:-len(last_token_str)]
            else:
                rec_prefix = current_prefix
            rec_prefix += token_str
            
            # if we did not extend the last token then we recurse while ignoring the possibility of extending the last token
            if token_str == last_token_str:
                sub_logprobs = await recursive_select(rec_prefix, allow_token_extension=False)
            else:
                sub_logprobs = await recursive_select(rec_prefix)
            
            # we add the logprob of this token to the logprob of the suffix
            for k in sub_logprobs:
                logprobs_out[k] = sub_logprobs[k] + logprob

        # if we did token healing and did not extend past our prefix we need to consider the next token
        # TODO: when returning all logprobs we need to consider all the options, which means we should
        # force the model to not token heal and see what would have happened then on the next token...
        # first_token_str = max(top_logprobs, key=top_logprobs.get)
        # if len(logprobs_result["top_logprobs"]) > 1 and len(first_token_str) == remove_prefix:            
        #     top_logprobs = logprobs_result["top_logprobs"][1]
        #     for token_str,logprob in top_logprobs.items():
        #         sub_logprobs = await recursive_select(current_prefix + token_str)
        #         for k in sub_logprobs:

        #             # compute the probability of a logical OR between the new extension and the previous possible ones
        #             p1 = np.exp(logprobs_out[k])
        #             p2 = np.exp(sub_logprobs[k] + logprob)
        #             or_prob = p1 + p2 - p1*p2
        #             logprobs_out[k] = np.log(or_prob)

        return logprobs_out
        
    # recursively compute the logprobs for each option
    option_logprobs = await recursive_select("")

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