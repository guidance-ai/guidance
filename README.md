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

When using `guidance`, you can work with large language models using Pythonic idioms:

```python
from guidance import system, user, assistant, gen
from guidance.models import Transformers

lm = Transformers("microsoft/Phi-4-mini-instruct")

with system():
    lm += "You are a helpful assistant"

with user():
    lm += "Hello. What is your name?"

with assistant():
    lm += gen(max_tokens=20)
```