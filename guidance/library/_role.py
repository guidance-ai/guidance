import guidance
from ._block import block

@guidance
def role_opener(lm, role_name, debug=False, **kwargs):
    if not hasattr(lm, "get_role_start"):
        raise Exception(f"You need to use a chat model in order the use role blocks like `with {role_name}():`! Perhaps you meant to use the {type(lm).__name__}Chat class?")
    lm += f"<||_html:<div style='display: flex; border-bottom: 1px solid rgba(127, 127, 127, 0.2);  justify-content: center; align-items: center;'><div style='flex: 0 0 80px; opacity: 0.5;'>{role_name.lower()}</div><div style='flex-grow: 1; padding: 5px; padding-top: 10px; padding-bottom: 10px; margin-top: 0px; white-space: pre-wrap; margin-bottom: 0px;'>_||>"
    
    if debug:
        lm += f"<||_html:<span style='background-color: rgba(255, 165, 0); border-radius: 3px;'>_||>"
    else:
        lm += "<||_#NODISP_||>"

    lm += lm.get_role_start(role_name, **kwargs)
    
    if debug:
        lm += "<||_html:</span>_||>"
    else:
        lm += "<||_/NODISP_||>"

    return lm

@guidance
def role_closer(lm, role_name, debug=False, **kwargs):

    if debug:
        lm += f"<||_html:<span style='background-color: rgba(255, 165, 0); border-radius: 3px;'>_||>"
    else:
        lm += "<||_#NODISP_||>"
    
    lm += lm.get_role_end(role_name)
    
    if debug:
        lm += "<||_html:</span>_||>"
    else:
        lm += "<||_/NODISP_||>"

    lm += "<||_html:</div></div>_||>"

    return lm

def role(role_name, text=None, **kwargs):
    if text is None:
        return block(opener=role_opener(role_name, **kwargs), closer=role_closer(role_name, **kwargs))
    else:
        assert False
        #return self.append(open_text + text + close_text)

def system(text=None, **kwargs):
    return role("system", text, **kwargs)

def user(text=None, **kwargs):
    return role("user", text, **kwargs)

def assistant(text=None, **kwargs):
    return role("assistant", text, **kwargs)

def function(text=None, **kwargs):
    return role("function", text, **kwargs)

def instruction(text=None, **kwargs):
    return role("instruction", text, **kwargs)