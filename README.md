<div align="right"><a href="https://guidance.readthedocs.org"><img src="https://readthedocs.org/projects/guidance/badge/?version=latest&style=flat" /></a></div>
<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/guidance_logo_blue_dark.svg">
  <img alt="guidance" src="docs/figures/guidance_logo_blue.svg" width=300">
</picture></div>
<br/>

**Guidance is an efficient programming paradigm for steering language models.** With Guidance, you can control how output is structured and get high-quality output for your use case—*while reducing latency and cost vs. conventional prompting or fine-tuning.* It allows users to constrain generation (e.g. with regex and CFGs) as well as to interleave control (conditionals, loops, tool use) and generation seamlessly.

   * [Install](#install)
   * [Features](#features)
   * [Example notebooks](#example-notebooks)
   * [Basic generation](#basic-generation)
   * [Constrained generation](#constrained-generation)
   * [Stateful control + generation](#stateful-control--generation)


## Install
Guidance is available through PyPI and supports a variety of backends (Transformers, llama.cpp, OpenAI, etc.). To use a specific model see [loading models](#loading-models).
```bash
pip install guidance
```

<!-- For a detailed walkthrough of using Guidance on hosted Phi models, check the [Azure AI specific loading instructions.](#azure-ai) and the [Phi-3 + Guidance cookbook](https://github.com/microsoft/Phi-3CookBook/blob/main/code/01.Introduce/guidance.ipynb). -->

<!-- <a href="https://www.youtube.com/watch?v=9oXjP5IIMzQ"  aria-label="Watch demo"><img alt="Watch demo" src="docs/figures/watch_demo_button.png" width="120"></a> <a href="#get-started" aria-label="Get started"><img alt="Watch demo" src="docs/figures/get_started_button.png" width="120"></a> -->

## Features

### A Pythonic interface for language models

When using Guidance, you can work with large language models using Pythonic idioms:

```python
from guidance import system, user, assistant, gen
from guidance.models import Transformers

# Could also do LlamaCpp or many other models
phi_lm = Transformers("microsoft/Phi-4-mini-instruct")

# Model objects are immutable, so this is a copy
lm = phi_lm

with system():
    lm += "You are a helpful assistant"

with user():
    lm += "Hello. What is your name?"

with assistant():
    lm += gen(max_tokens=20)

print(lm)
```

It's also really easy to capture generated text:

```python
# Get a new copy of the Model
lm = phi_lm

with system():
    lm += "You are a helpful assistant"

with user():
    lm += "Hello. What is your name?"

with assistant():
    lm += gen(name="lm_response", max_tokens=20)

print(f"{lm['lm_response']=}")
```

### Guarantee output syntax with constrained generation

Guidance provides an easy to use, yet immensely powerful syntax for constraining the output of a language model.
For example, a `gen()` call can be constrained to match a regular expression:

```python
lm = phi_lm

with system():
    lm += "You are a teenager"

with user():
    lm += "How old are you?"

with assistant():
    lm += gen("lm_age", regex=r"\d+", temperature=0.8)

print(f"The language model is {lm['lm_age']} years old")
```

Often, we know that the output has to be an item from a list we know in advance.
Guidance provides a `select()` function for this scenario:

```python
from guidance import select

lm = phi_lm

with system():
    lm += "You are a geography expert"

with user():
    lm += """What is the capital of Sweden? Answer with the correct letter.

    A) Helsinki
    B) Reykjavík 
    C) Stockholm
    D) Oslo
    """

with assistant():
    lm += select(["A", "B", "C", "D"], name="model_selection")

print(f"The model selected {lm['model_selection']}")
```

The constraint system offered by Guidance is extremely powerful.
It can ensure that the output conforms to any context free grammar (so long as the backend LLM has full support for Guidance).
More on this below.

### Create your own Guidance functions

With Guidance, you can create your own Guidance functions which can interact with language models.
These are marked using the `@guidance` decorator.
Suppose we wanted to answer lots of multiple choice questions.
We could do something like the following:

```python
import guidance

from guidance.models import Model

ASCII_OFFSET = ord("a")

@guidance
def zero_shot_multiple_choice(
    language_model: Model,
    question: str,
    choices: list[str],
):
    with user():
        language_model += question + "\n"
        for i, choice in enumerate(choices):
            language_model += f"{chr(i+ASCII_OFFSET)} : {choice}\n"

    with assistant():
        language_model += select(
            [chr(i + ASCII_OFFSET) for i in range(len(choices))], name="string_choice"
        )

    return language_model
```
Now, define some questions:
```python
questions = [
    {
        "question" : "Which state has the northernmost capital?",
        "choices" : [
            "New South Wales",
            "Northern Territories",
            "Queensland",
            "South Australia",
            "Tasmania",
            "Victoria",
            "Western Australia",
        ],
        "answer" : 1,
    },
    {
        "question" : "Which of the following is venomous?",
        "choices" : [
            "Kangaroo",
            "Koala Bear",
            "Platypus",
        ],
        "answer" : 2,
    }
]
```
We can use our decorated function like `gen()` or `select()`.
The `language_model` argument will be filled in for us automatically:
```python
lm = phi_lm

with system():
    lm += "You are a student taking a multiple choice test."

for mcq in questions:
    lm_temp = lm + zero_shot_multiple_choice(question=mcq["question"], choices=mcq["choices"])
    converted_answer = ord(lm_temp["string_choice"]) - ASCII_OFFSET
    print(lm_temp)
    print(f"LM Answer: {converted_answer},  Correct Answer: {mcq['answer']}")
```

### Generating JSON

A JSON schema is actually a context free grammar, and hence it can be used to constrain an LLM using Guidance.
This is a common enough case that Guidance provides special support for it.
A quick sample, based on a Pydantic model:
```python
import json
from pydantic import BaseModel, Field

from guidance import json as gen_json

class BloodPressure(BaseModel):
    systolic: int = Field(gt=0, le=300)
    diastolic: int = Field(gt=0, le=200)
    location: str = Field(max_length=50)
    model_config = dict(extra="forbid")

lm = phi_lm

with system():
    lm += "You are a doctor taking a patient's blood pressure taken from their arm"

with user():
    lm += "Report the blood pressure"

with assistant():
    lm += gen_json(name="bp", schema=BloodPressure)

print(f"{lm['bp']=}")

# Use Python's JSON library
loaded_json = json.loads(lm["bp"])
print(json.dumps(loaded_json, indent=4))

# Use Pydantic
result = BloodPressure.model_validate_json(lm["bp"])
print(result.model_dump_json(indent=8))
```