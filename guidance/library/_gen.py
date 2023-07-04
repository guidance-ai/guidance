import types

import guidance
from guidance import InPlace

@guidance
def gen(lm, name=None, *, max_tokens=10, list_append=False, pattern=None, stop=None, stop_regex=None, n=1, temperature=0.0, top_p=1.0,
        logprobs=None, cache_seed=None, token_healing=None, stream=None, function_call="none", save_stop_text=False, **llm_kwargs):

    # set stream if we are interactive
    if stream is None and not lm.silent:
        stream = True

    new_lm = lm._clone()

    with InPlace(lm) as new_lm:
        new_lm += f"<||_html:<span style='background-color: rgba(0, 165, 0, 0.25)'>_||>"

        if name is not None:
            new_lm[name] = ""

        gen_obj = lm.session(
            str(lm), stop=stop, stop_regex=stop_regex, max_tokens=max_tokens, n=n, pattern=pattern,
            temperature=temperature, top_p=top_p, logprobs=logprobs, cache_seed=cache_seed, token_healing=token_healing,
            echo=getattr(lm, "logprobs", False), stream=stream, function_call=function_call, **llm_kwargs
        )

        if not isinstance(gen_obj, (types.GeneratorType, list, tuple)):
            gen_obj = [gen_obj]

        if n == 1:
            generated_value = ""
            logprobs_out = []
            if list_append:
                lm[name] = lm.get(name, [])
                lm[name].append("")
                list_ind = len(lm[name])-1
                if logprobs is not None:
                    lm[name+"_logprobs"] = lm.get(name+"_logprobs", [])
                    lm[name+"_logprobs"].append([])
                    assert len(len(lm[name])) == len(len(lm[name+"_logprobs"]))
            for resp in gen_obj:
                new_text = resp["choices"][0].get("text", "")
                generated_value += new_text
                new_lm += new_text
                if logprobs is not None:
                    logprobs_out.extend(resp["choices"][0]["logprobs"]["top_logprobs"])
                if list_append:
                    lm[name][list_ind] = generated_value
                    if logprobs is not None:
                        lm[name+"_logprobs"][list_ind] = logprobs_out
                elif name is not None:
                    lm[name] = generated_value
                    if logprobs is not None:
                        lm[name+"_logprobs"] = logprobs_out
            
            # save the final stopping text if requested
            if save_stop_text is not False:
                if save_stop_text is True:
                    save_stop_text = name+"_stop_text"
                lm[save_stop_text] = resp["choices"][0].get('stop_text', None)
            
            if list_append:
                lm[name][list_ind] = generated_value
            elif name is not None:
                lm[name] = generated_value

        # for c in rand_string:
        #     time.sleep(0.1) # simulate a long-running task
        #     new_lm += c
        #     if name is not None:
        #         new_lm[name] += c
        
        new_lm += f"<||_html:</span>_||>"
    
    return new_lm