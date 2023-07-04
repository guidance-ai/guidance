import json

def function_def(function):
    ''' Dumps a function definition in a format that matches what the gen command expects with stopping at function calls.
    '''

    # if it is a function we extract the parameters and docs
    if callable(function):
        import inspect
        parameters = inspect.signature(function).parameters
        docs = inspect.getdoc(function)
        if docs is None:
            docs = ""
        out = {
            "name": function.__name__,
            "description": docs.split("\n")[0].strip(),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

        for p in parameters:
            out["parameters"]["properties"][p] = {
                "type": "string"
            }
            out["parameters"]["required"].append(p)

    elif isinstance(function, dict):
        return json.dumps(function)
