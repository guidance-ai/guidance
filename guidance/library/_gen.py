import asyncio
import re
import uuid
from .._grammar import grammar

async def gen(variable_name="generated", partial_output=None, parse=False, stop=None, max_tokens=500, n=1, temperature=0.0, top_p=1.0, logprobs=None, hidden=False, parser_prefix=None, parser=None, prefix="", suffix="", next_text=None, prev_text=None, **kwargs):
    ''' Use the LM to generate a completion string that is stored in the variable `variable_name`.
    '''

    # if stop is None then we use the text of the node after the generate command
    if stop is None:
        if next_text is not None and prev_text is not None:

            # auto-detect quote stop tokens
            quote_types = ['"', "'", "'''", '"""', "`"]
            for quote_type in quote_types:
                if next_text.startswith(quote_type) and prev_text.endswith(quote_type):
                    stop = quote_type
                    break
                    
            # auto-detect XML tag stop tokens
            if stop is None:
                m = re.match(r"<([^>]+)>", next_text)
                if m is not None:
                    end_tag = "</"+m.group(1)+">"
                    if next_text.startswith(end_tag):
                        stop = end_tag
                else:
                    stop = next_text
                
        else:
            stop = next_text
    
    # set the cache seed to 0 if temperature is 0
    if temperature > 0:
        cache_seed = parser.program.cache_seed
        parser.program.cache_seed += 1
    else:
        cache_seed = 0

    # see if we should stream the results
    if n == 1:
        stream_generation = parser.program.stream is True or (parser.program.stream is None and parser.program.echo is True)
    else:
        stream_generation = False

    # call the LLM
    gen_obj = parser.program.llm(
        parser_prefix+prefix, stop=stop, max_tokens=max_tokens, n=n,
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

        # TODO: we could enable the parsing to branch into multiple paths here, but for now we just complete the program with the first prefix
        generated_value = generated_values[0]

        # echoing with multiple completions is not standard behavior
        # this just uses the first generated value for completion and the rest as alternatives only used for the variable storage
        # we mostly support this so that the echo=False hiding behavior does not make multiple outputs more complicated than it needs to be in the UX
        # if echo:
        #     partial_output(generated_value) 
        
        id = uuid.uuid4().hex
        l = len(generated_values)
        out = "{{!--" + f"GMARKER_generate_many_start_{not hidden}_{l}${id}$" + "--}}"
        for i, value in enumerate(generated_values):
            if i > 0:
                out += "{{!--" + f"GMARKER_generate_many_{not hidden}_{i}${id}$" + "--}}"
            out += value
        partial_output(out + "{{!--" + f"GMARKER_generate_many_end${id}$" + "--}}")
        return
        # return "{{!--GMARKER_generate_many_start$$}}" + "{{!--GMARKER_generate_many$$}}".join([v for v in generated_values]) + "{{!--GMARKER_generate_many_end$$}}"
        # return "".join([v for v in generated_values])