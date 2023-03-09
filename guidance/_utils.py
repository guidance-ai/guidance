import os
import requests
import inspect

def load(guidance_file):
    ''' Load a guidance prompt from the given text file.

    If the passed file is a valid local file it will be loaded directly.
    Otherwise, if it starts with "http://" or "https://" it will be loaded
    from the web.
    '''

    if os.path.exists(guidance_file):
        with open(guidance_file, 'r') as f:
            return f.read()
    elif guidance_file.startswith('http://') or guidance_file.startswith('https://'):
        return requests.get(guidance_file).text
    else:
        raise ValueError('Invalid guidance file: %s' % guidance_file)
    
def chain(prompts, **kwargs):
    ''' Chain together multiple prompts into a single prompt.
    
    This merges them into a single prompt like: {{>prompt1 hidden=True}}{{>prompt2 hidden=True}}
    '''

    from ._prompt import Prompt

    new_template = "".join(["{{>prompt%d hidden=True}}" % i for i in range(len(prompts))])
    for i, prompt in enumerate(prompts):
        if isinstance(prompt, Prompt):
            kwargs["prompt%d" % i] = prompt
        else:
            sig = inspect.signature(prompt)
            args = ""
            for name,_ in sig.parameters.items():
                args += f" {name}={name}"
            fname = find_func_name(prompt, kwargs)
            kwargs["prompt%d" % i] = Prompt("{{set (%s%s)}}" % (fname, args), **{fname: prompt})
            # kwargs.update({f"func{i}": prompt})
    return Prompt(new_template, **kwargs)

def find_func_name(f, used_names):
    if hasattr(f, "__name__"):
        prefix = f.__name__.replace("<", "").replace(">", "")
    else:
        prefix = "function"
    
    if prefix not in used_names:
        return prefix
    else:
        for i in range(100):
            fname = f"{prefix}{i}"
            if fname not in used_names:
                return fname