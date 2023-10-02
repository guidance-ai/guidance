import pygtrie
import numpy as np
from .._utils import ContentCapture


async def selectm(
    variable_name="selected",
    options=None,
    logprobs=None,
    _parser_context=None,
    sep=None,
):
    """Select multiple values from a list of choices.

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
    """
    parser = _parser_context["parser"]
    block_content = _parser_context["block_content"]
    variable_stack = _parser_context["variable_stack"]
    next_node = _parser_context["next_node"]
    next_next_node = _parser_context["next_next_node"]

    if block_content is None:
        assert (
            options is not None
        ), "You must provide an options list like: {{select 'variable_name' options}} when using the select command in non-block mode."
    else:
        assert (
            len(block_content) > 1
        ), "You must provide at least one two options to the select block command."
        assert (
            options is None
        ), "You cannot provide an options list when using the select command in block mode."

    assert (
        sep is not None or sep == ""
    ), "You must provide a separator to the selectm command."

    if options is None:
        options = [str(block_content[0].content[0])]
        for i in range(1, len(block_content), 2):
            assert (
                block_content[i][0] == "or"
            ), "You must provide a {{or}} between each option in a select block."
            options.append(
                str(block_content[i + 1].content[0])
            )  # block_content[i+1].text)

    # find what text follows the select command and append it to the options.
    # we do this so we can differentiate between select options where one is a prefix of another
    next_text = next_node[0] if next_node is not None else ""
    if next_next_node and next_next_node.text.startswith("{{~"):
        next_text = next_text.lstrip()
        if next_next_node and next_text == "":
            next_text = next_next_node.text
    if (
        next_text == ""
    ):  # if we have nothing after us then we are at the end of the text
        next_text = parser.program.llm.end_of_text()

    while len(options) > 1:
        options_aux = [option + next_text for option in options]
        options_aux += [option + sep for option in options]

        # TODO: this retokenizes the whole prefix many times, perhaps this could become a bottleneck?
        options_tokens = [
            parser.program.llm.encode(variable_stack["@prefix"] + option)
            for option in options_aux
        ]

        # encoding the prefix and then decoding it might change the length, so we need to account for that
        recoded_parser_prefix_length = len(
            parser.program.llm.decode(
                parser.program.llm.encode(variable_stack["@prefix"])
            )
        )

        # build a trie of the options
        token_map = pygtrie.Trie()
        for i, option in enumerate(options_tokens):
            token_map[option] = i

        async def recursive_select(current_prefix, allow_token_extension=True):
            """Returns a dictionary of scores for each option (keyed by the option index)."""

            def get_extension_options(prefix):
                """Get possible extension options for a given prefix."""
                try:
                    return token_map.items(prefix=prefix)
                except KeyError:
                    return []

            def compute_logprob_or(logprob1, logprob2):
                """Compute the combined probability for two log probabilities."""
                p1 = np.exp(logprob1)
                p2 = np.exp(logprob2)
                combined_prob = p1 + p2 - p1 * p2
                return np.log(combined_prob)

            # Find which select options are possible
            extension_options = get_extension_options(current_prefix)

            # Return early if no options
            if not extension_options:
                return {}

            # Single option scenario
            if len(extension_options) == 1:
                return {extension_options[0][0]: 0}

            # Determine the longest common prefix among the valid options
            match_index = len(current_prefix)
            for i in range(
                len(current_prefix), min([len(o[0]) for o in extension_options])
            ):
                if len(set([o[0][i] for o in extension_options])) > 1:
                    break
                match_index += 1

            # Update current prefix with the common match
            if match_index > len(current_prefix):
                current_prefix += extension_options[0][0][
                    len(current_prefix) : match_index
                ]

            # Bias the logits towards valid options
            logit_bias = {
                option_tokens[match_index]: 100
                for option_tokens, index in extension_options
            }

            # Check for where we are at the end of the prefix
            logprobs_out = {option[0]: -1000 for option in extension_options}
            if not logit_bias and current_prefix in [o[0] for o in extension_options]:
                logprobs_out[current_prefix] = 0
                return logprobs_out

            # Generate the token logprobs
            gen_obj = await parser.llm_session(
                parser.program.llm.decode(current_prefix),
                max_tokens=1,
                logit_bias=logit_bias,
                logprobs=len(logit_bias),
                cache_seed=0,
                token_healing=False,
            )

            # Process LLM output
            if "logprobs" in gen_obj["choices"][0]:
                logprobs_result = gen_obj["logprobs"]

                # convert the logprobs keys from string back to token ids
                top_logprobs = {}
                for k, v in logprobs_result["top_logprobs"][0].items():
                    id = parser.program.llm.token_to_id(k)
                    top_logprobs[id] = v
            else:
                assert (
                    logprobs is None
                ), "You cannot ask for the logprobs in a select call when using a model that does not return logprobs!"
                top_logprobs = {parser.program.llm.token_to_id(gen_obj["text"]): 0}

            # no need to explore all branches if we are just taking the greedy max
            if logprobs is None:
                max_key = max(top_logprobs, key=top_logprobs.get)
                top_logprobs = {max_key: top_logprobs[max_key]}

            # Recursively call for possible extensions
            for token, logprob in top_logprobs.items():
                sub_logprobs = await recursive_select(current_prefix + [token])

                # Update log probabilities
                for k, v in sub_logprobs.items():
                    logprobs_out[k] = compute_logprob_or(
                        logprobs_out.get(k, -1000), v + logprob
                    )

            return logprobs_out

        # recursively compute the logprobs for each option
        option_logprobs = await recursive_select([])

        # convert the key from a token list to a string
        option_logprobs = {
            parser.program.llm.decode(k): v for k, v in option_logprobs.items()
        }

        # trim off the prefix and suffix we added to the options
        option_logprobs = {
            k[recoded_parser_prefix_length:]: v for k, v in option_logprobs.items()
        }

        # select the option with the highest logprob
        selected_option = max(option_logprobs, key=option_logprobs.get)

        value_list = variable_stack.get(variable_name, [])
        value_list.append(selected_option)
        if logprobs is not None:
            variable_stack[logprobs] = option_logprobs

        if max(option_logprobs.values()) <= -1000:
            raise ValueError(
                "No valid option generated in #select! Please post a GitHub issue since this should not happen :)"
            )

        if selected_option.endswith(next_text):
            variable_stack["@prefix"] += selected_option[: -len(next_text)]
            break
        variable_stack["@prefix"] += selected_option
        options.remove(selected_option[: -len(sep)])


selectm.is_block = True
