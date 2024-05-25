from .._guidance import guidance
from .._grammar import capture as grammar_capture, GrammarFunction

# Adapted from active_role_end in _model.py, functionality should be shared probably?
import re
format_pattern = re.compile(r"<\|\|_.*?_\|\|>", flags=re.DOTALL)

@guidance(stateless=lambda *args, **kwargs: isinstance(args[0], GrammarFunction))
def capture(lm, value, name):
    if isinstance(value, GrammarFunction):
        return lm + grammar_capture(value, name)
    else:
        start_len = len(lm)
        lm += value
        # Adapted from active_role_end in _model.py
        parts = ""
        for _, role_end_str in lm.opened_blocks.values():
            role_end_str = format_pattern.sub("", role_end_str)
            if len(role_end_str) > 0 and not re.fullmatch(r"\s+", role_end_str):
                parts += role_end_str
        
        return lm.set(name, str(lm)[start_len-len(parts):].removesuffix(parts))