def set(name, value=None, hidden=True, _parser_context=None):
    ''' Set the value of a variable or set of variables.

    Parameters
    ----------
    name : str or dict
        If a string, the name of the variable to set. If a dict, the keys are the variable names and the values are the values to set.
    value : str, optional
        The value to set the variable to. Only used if `name` is a string.
    hidden : bool, optional
        If True, the variable will be set but not printed in the output.
    '''
    parser = _parser_context['parser']
    variable_stack = _parser_context['variable_stack']

    if not parser.executing:
        return ""

    if isinstance(name, dict):
        assert hidden is not False, "hidden cannot be False if setting multiple variables!"
        for k, v in name.items():
            variable_stack[k] = v
        out = ""
        for k, v in name.items():
            if isinstance(v, str):
                if "\n" in v:
                    v = f'"""{v}"""'
                elif '"' in v:
                    v = f"'{v}'"
                else:
                    v = f'"{v}"'
            out += f" {k}={v}"
        out += ""
        return "{{!--GMARKER_set$" + out + "$--}}"
    else:
        variable_stack[name] = value
        if hidden is not True:
            return value
        else:
            out = "{{set "+name+"=" + str(value) + "}}"
            return "{{!--GMARKER_set$" + out + "$--}}"