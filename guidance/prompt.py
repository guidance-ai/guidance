import re
from . import generators

class Prompt:
    def __init__(self, prompt, generator=None):
        self.prompt = prompt
        self.generator = generator

        # default to an OpenAI generator
        if self.generator is None:
            self.generator = generators.OpenAI()
    
    def __call__(self, variables, execution_method="fixed_prefix"):

        if execution_method == "fixed_prefix":
            out = parse(self.prompt, variables)
            fixed_prefix = re.sub(r'\{\{get .*', '', out, flags=re.DOTALL)
            output_template_escaped = re.escape(out[len(fixed_prefix):])

            out2 = self.generator(fixed_prefix, max_tokens=250)

            pattern = re.sub(r'\\\{\\\{get\\ ([^\\]+)\\\}\\\}', r'(?P<\1>.*)', output_template_escaped)
            match = re.match(pattern, out2["choices"][0]["text"], flags=re.DOTALL)
            return match.groupdict()


def parse(prompt, variables={}):
    # print("prompt", prompt)
    tag_open = False
    tag_start = 0
    out = ""
    i = 0
    in_recurse_group = False
    recurse_group_start = 0
    recurse_group_name = ""
    recurse_group_args = []
    recurse_group_depth = 0
    for_item_name = ""

    # walk through the prompt character by character
    while i < len(prompt):

        # find tag starts
        if prompt[i:i+2] == '{{':
            tag_open = True
            tag_start = i+2
            i += 2

        # process tags on tag ends
        elif prompt[i:i+2] == '}}':
            tag_open = False
            # tag_end = i
            tag_name = prompt[tag_start:i]
            i += 2

            # if we are in a recurse group then we need to just skip over all the internal content to find the closing tag
            if in_recurse_group:
                if tag_name.startswith("#"):
                    raw_name = tag_name.split()[0][1:]
                    if raw_name == recurse_group_name:
                        recurse_group_depth += 1
                elif tag_name.startswith("/"):
                    raw_name = tag_name[1:]
                    if raw_name == recurse_group_name:
                        recurse_group_depth -= 1
                        if recurse_group_depth == 0:
                            if raw_name == "each":
                                items = variables.get(recurse_group_args[0], [])
                                for j, var in enumerate(items):
                                    # print("VAR", var)
                                    out += parse(prompt[recurse_group_start:tag_start-2], variables=variables | {"this": var} | {"@last": j == len(items)-1, "@first": j == 0, "@index": j})
                            elif raw_name == "for":
                                assert recurse_group_args[1] == "in"
                                items = variables.get(recurse_group_args[2], [])
                                item_name = recurse_group_args[0]
                                for j, var in enumerate(items):
                                    # print("VAR", var)
                                    out += parse(prompt[recurse_group_start:tag_start-2], variables=variables | {item_name: var} | {"@last": j == len(items)-1, "@first": j == 0, "@index": j})
                            elif raw_name == "if":
                                if variables.get(recurse_group_args[0], False):
                                    out += parse(prompt[recurse_group_start:tag_start-2], variables=variables)
                            elif raw_name == "unless":
                                if not variables.get(recurse_group_args[0], False):
                                    out += parse(prompt[recurse_group_start:tag_start-2], variables=variables)
                            in_recurse_group = False
                            recurse_group_name = ""
                            recurse_group_start = 0
            
            # if we are not in a recurse group then we need to process the tag
            else:
                if tag_name.startswith("#"):
                    raw_name = tag_name.split()[0][1:]
                    in_recurse_group = True
                    recurse_group_start = i
                    recurse_group_name = raw_name
                    recurse_group_args = tag_name.split()[1:]
                    recurse_group_depth = 1
                    
                elif tag_name.startswith("/"):
                    raise Exception("Closing tag without opening tag:", tag_name)
                else:
                    parts = tag_name.split()
                    if len(parts) == 1:
                        if "." in tag_name:
                            var_name, var_attr = tag_name.split(".", 2)
                            out += str(variables.get(var_name, {}).get(var_attr, ""))
                        else:
                            out += str(variables.get(tag_name, ""))
                    elif len(parts) == 2:
                        if parts[0] == "get":
                            # print("Call the LM with:\n", out)
                            out += "{{" + tag_name + "}}" # TODO: need to record so we can enable later dependencies on the answer
            # print("===", tag_name)
        elif not tag_open and not in_recurse_group:
            out += prompt[i]
            i += 1
        else:
            i += 1

    return out