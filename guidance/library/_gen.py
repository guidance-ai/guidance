import asyncio
import re
import uuid
import logging
from .._grammar import grammar

log = logging.getLogger(__name__)

async def gen(variable_name="generated", partial_output=None, parse=False, stop=None, stop_regex=None, max_tokens=500, n=1, temperature=0.0, top_p=1.0, logprobs=None, pattern=None, hidden=False, save_prompt=False, parser_prefix=None, parser=None, prefix="", suffix="", next_node=None, prev_node=None, next_next_node=None, **kwargs):
    ''' Use the LM to generate a completion string that is stored in the variable `variable_name`.
    '''

    # if stop is None then we use the text of the node after the generate command
    if stop is None:

        next_text = next_node.text if next_node is not None else ""
        prev_text = prev_node.text if prev_node is not None else ""
        if next_next_node and next_next_node.text.startswith("{{~"):
            next_text = next_text.lstrip()
            if next_next_node and next_text == "":
                next_text = next_next_node.text

        # auto-detect role stop tags
        m = re.match(r"{{~?/(user|assistant|system|role)~?}}", next_text)
        if m:
            stop = parser.program.llm.role_end(m.group(1))

        # auto-detect quote stop tokens
        quote_types = ['"', "'", "'''", '"""', "`"]
        for quote_type in quote_types:
            if next_text.startswith(quote_type) and prev_text.endswith(quote_type):
                stop = quote_type
                break
                
        # auto-detect XML tag stop tokens
        if stop is None:
            m = re.match(r"<([^>\W]+)[^>]+>", next_text)
            if m is not None:
                end_tag = "</"+m.group(1)+">"
                if next_text.startswith(end_tag):
                    stop = end_tag
            else:
                stop = next_text
    if stop == "":
        stop = None

    # set the cache seed to 0 if temperature is 0
    if temperature > 0:
        cache_seed = parser.program.cache_seed
        parser.program.cache_seed += 1
    else:
        cache_seed = 0

    # see if we should stream the results
    if n == 1:
        stream_generation = parser.program.stream is True or parser.program.echo#_displaying # (parser.program.stream is None and parser.program.echo is True)
    else:
        stream_generation = False

    # save the prompt if requested
    if save_prompt:
        parser.set_variable(save_prompt, parser_prefix+prefix)

    # call the LLM
    gen_obj = parser.llm_session(
        parser_prefix+prefix, stop=stop, stop_regex=stop_regex, max_tokens=max_tokens, n=n, pattern=pattern,
        temperature=temperature, top_p=top_p, logprobs=parser.program.logprobs, cache_seed=cache_seed,
        echo=parser.program.logprobs is not None, stream=stream_generation
    )

    if n == 1:
        generated_value = prefix
        partial_output(prefix)
        logprobs_out = []
        if not stream_generation:
            gen_obj = [gen_obj]
        for resp in gen_obj:
            await asyncio.sleep(0) # allow other tasks to run
            #log("parser.should_stop = " + str(parser.should_stop))
            if parser.should_stop:
                #log("Stopping generation")
                break
            # log.debug("resp", resp)
            generated_value += resp["choices"][0]["text"]
            partial_output(resp["choices"][0]["text"])
            if logprobs is not None:
                logprobs_out.extend(resp["choices"][0]["logprobs"])
            parser.set_variable(variable_name, generated_value)
            if logprobs is not None:
                parser.set_variable(variable_name+"_logprobs", logprobs_out)
        if hasattr(gen_obj, 'close'):
            gen_obj.close()
        generated_value += suffix
        partial_output(suffix)
        parser.set_variable(variable_name, generated_value)
        
        if parse:
            assert not hidden, "Cannot parse generated text if we are hiding the output" # TODO: fix this?
            subtree = grammar.parse(generated_value)
            return await parser.visit(subtree)
        else:
            # stop executing if we were interrupted
            if parser.should_stop:
                parser.executing = False
                parser.should_stop = False
            return
    else:
        assert not isinstance(gen_obj, list), "Streaming is only supported for n=1"
        generated_values = [prefix+choice["text"]+suffix for choice in gen_obj["choices"]]
        parser.set_variable(variable_name, generated_values)
        if logprobs is not None:
            parser.set_variable(variable_name+"_logprobs", [choice["logprobs"] for choice in gen_obj["choices"]])

        if not hidden:
            # TODO: we could enable the parsing to branch into multiple paths here, but for now we just complete the program with the first prefix
            generated_value = generated_values[0]

            # echoing with multiple completions is not standard behavior
            # this just uses the first generated value for completion and the rest as alternatives only used for the variable storage
            # we mostly support this so that the echo=False hiding behavior does not make multiple outputs more complicated than it needs to be in the UX
            # if echo:
            #     partial_output(generated_value) 
            
            id = uuid.uuid4().hex
            l = len(generated_values)
            out = "{{!--" + f"GMARKERmany_generate_start_{not hidden}_{l}${id}$" + "--}}"
            for i, value in enumerate(generated_values):
                if i > 0:
                    out += "{{!--" + f"GMARKERmany_generate_{not hidden}_{i}${id}$" + "--}}"
                out += value
            partial_output(out + "{{!--" + f"GMARKERmany_generate_end${id}$" + "--}}")
            return
            # return "{{!--GMARKERmany_generate_start$$}}" + "{{!--GMARKERmany_generate$$}}".join([v for v in generated_values]) + "{{!--GMARKERmany_generate_end$$}}"
            # return "".join([v for v in generated_values])