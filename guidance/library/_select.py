import random
import time

import guidance
from guidance import InPlace

@guidance
def select(lm, name=None, *, options):
    with InPlace(lm) as new_lm:
        new_lm += f"<||_html:<span style='background-color: rgba(0, 165, 0, 0.25)'>_||>"
        selected = random.choice(options)
        time.sleep(0.5) # simulate a long-running task
        new_lm += selected
        if name is not None:
            new_lm[name] = selected
        new_lm += f"<||_html:</span>_||>"
    return new_lm