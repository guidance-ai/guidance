<div align="right">
  <a href="https://discord.gg/cjPfAK43dz"><img src="https://img.shields.io/badge/Discord-Join%20Us-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="mailto:guidanceai@microsoft.com"><img src="https://img.shields.io/badge/Email-guidanceai%40microsoft.com-0078D4?logo=microsoft-outlook&logoColor=white" alt="Email"></a>
  <img src="https://img.shields.io/badge/Hours-10am--2pm%20Pacific-gray" alt="Hours">
</div>
<div align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/guidance_logo_blue_dark.svg">
  <img alt="guidance" src="docs/figures/guidance_logo_blue.svg" width=300">
</picture></div>
<br/>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

**Guidance 是一种用于驾驭和引导大语言模型（LLM）的高效编程范式。** 借助 Guidance，您可以精确控制输出的结构，并针对您的业务场景获得高质量的生成结果——*同时相比传统的提示词工程（Prompting）或微调（Fine-tuning），显著降低延迟与计算成本。* 它不仅允许用户对生成过程施加约束（例如使用正则表达式和上下文无关文法 CFG），还能将程序控制流（条件分支、循环、工具调用）与文本生成无缝交织在一起。

   * [安装](#安装)
   * [核心特性](#核心特性)


## 安装
Guidance 已发布至 PyPI，并支持多种后端（Transformers、llama.cpp、OpenAI 等）。
如果您已经安装了模型所需的后端，只需运行以下命令即可：
```bash
pip install guidance
```

## 核心特性

### 面向语言模型的 Python 风格（Pythonic）接口

使用 Guidance 时，您可以直接使用符合 Python 习惯的语法与大语言模型进行交互：

```python
from guidance import system, user, assistant, gen
from guidance.models import Transformers

# 也可以使用 LlamaCpp 或许多其他模型后端
phi_lm = Transformers("microsoft/Phi-4-mini-instruct")

# 模型对象是不可变的（immutable），因此这里会创建一个副本
lm = phi_lm

with system():
    lm += "You are a helpful assistant"

with user():
    lm += "Hello. What is your name?"

with assistant():
    lm += gen(max_tokens=20)

print(lm)
```
如果在命令行中运行，将产生如下输出：
```
<|system|>You are a helpful assistant<|end|><|user|>Hello. What is your name?<|end|><|assistant|>I am Phi, an AI developed by Microsoft. How can I help you today?
```
而如果在 Jupyter Notebook 中运行，Guidance 则会提供一个交互式小部件（Widget），带来更丰富的用户体验：

<img src="docs/figures/widget_basic_example_20250703.png" alt="Guidance widget showing HTML generation" />

使用 Guidance，捕获生成的文本非常轻松：

```python
# 获取模型的一个新副本
lm = phi_lm

with system():
    lm += "You are a helpful assistant"

with user():
    lm += "Hello. What is your name?"

with assistant():
    lm += gen(name="lm_response", max_tokens=20)

print(f"{lm['lm_response']=}")
```

```
lm['lm_response']='I am Phi, an AI developed by Microsoft. How can I help you today?'
```

### 通过约束生成保证输出语法

Guidance 提供了一套既易于使用又极其强大的语法，用于约束语言模型的输出。
例如，您可以约束一次 `gen()` 调用使其必须匹配指定的正则表达式：

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

```
The language model is 13 years old
```

在很多场景下，我们预先知道输出必须是某个已知列表中的一项。
Guidance 为此提供了 `select()` 函数：

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

```
The model selected C
```

Guidance 提供的约束系统极为强大。
它可以确保输出严格符合任何上下文无关文法（Context-Free Grammar，只要后端 LLM 完整支持 Guidance）。
下文将对此展开详细介绍。

### 离线调试文法（无需调用模型 API）

在迭代调试约束规则时，您可以在本地直接验证候选字符串，并使用 `Mock` 模型测试完整的运行流程。

```python
from guidance import gen
from guidance.models import Mock

grammar = "expr=" + gen(regex=r"\d+([+*]\d+)*", name="expr")

# 1) 直接针对文法验证字符串
assert grammar.match("expr=12+7*3") is not None
assert grammar.match("expr=12+*3") is None

# 2) 使用本地 Mock 模型运行相同的文法
lm = Mock(b"<s>expr=12+7*3")
lm += grammar
print(lm["expr"])  # 12+7*3
```

### 创建自定义的 Guidance 函数

借助 Guidance，您可以创建能够与语言模型交互的自定义 Guidance 函数。
这些函数使用 `@guidance` 装饰器进行标记。
假设我们想要回答大量多项选择题，可以编写如下代码：

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
现在，定义一些题目：
```python
questions = [
    {
        "question" : "Which state has the northernmost capital?",
        "choices" : [
            "New South Wales",
            "Northern Territory",
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
我们可以像使用 `gen()` 或 `select()` 一样使用被装饰的函数。
其中 `language_model` 参数会自动为我们填入：
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

```
<|system|>You are a student taking a multiple choice test.<|end|><|user|>Which state has the northernmost capital?
a : New South Wales
b : Northern Territory
c : Queensland
d : South Australia
e : Tasmania
f : Victoria
g : Western Australia
<|end|><|assistant|>b
LM Answer: 1,  Correct Answer: 1
<|system|>You are a student taking a multiple choice test.<|end|><|user|>Which of the following is venomous?
a : Kangaroo
b : Koala Bear
c : Platypus
<|end|><|assistant|>c
LM Answer: 2,  Correct Answer: 2
```

Guidance 函数可以相互组合，从而构建出完整的上下文无关文法（CFG）。
例如，我们可以创建一组 Guidance 函数来构建一个简单的 HTML 网页（注意：这里并非 HTML 的完整实现）。
我们首先编写一个生成不包含任何 HTML 标签的纯文本的简单函数。
该函数被标记为 `stateless=True`，表示我们打算将其用于组合文法：

```python
@guidance(stateless=True)
def _gen_text(lm: Model):
    return lm + gen(regex="[^<>]+")
```

随后，我们可以使用该函数在任意 HTML 标签内部生成文本：
```python
@guidance(stateless=True)
def _gen_text_in_tag(lm: Model, tag: str):
    lm += f"<{tag}>"
    lm += _gen_text()
    lm += f"</{tag}>"
    return lm
```
现在，我们来创建页面头部（header）。作为其中的一部分，我们需要生成一个页面标题（title）：
```python
@guidance(stateless=True)
def _gen_header(lm: Model):
    lm += "<head>\n"
    lm += _gen_text_in_tag("title") + "\n"
    lm += "</head>\n"
    return lm
```
HTML 页面的主体（body）将由各级标题（heading）和段落（paragraph）填充。
我们可以分别为它们定义生成函数：
```python
from guidance.library import one_or_more

@guidance(stateless=True)
def _gen_heading(lm: Model):
    lm += select(
        options=[_gen_text_in_tag("h1"), _gen_text_in_tag("h2"), _gen_text_in_tag("h3")]
    )
    lm += "\n"
    return lm

@guidance(stateless=True)
def _gen_para(lm: Model):
    lm += "<p>"
    lm += one_or_more(
        select(
            options=[
                _gen_text(),
                _gen_text_in_tag("em"),
                _gen_text_in_tag("strong"),
                "<br />",
            ],
        )
    )
    lm += "</p>\n"
    return lm
```
接着，编写用于定义 HTML 主体（body）本身的函数：
```python
@guidance(stateless=True)
def _gen_body(lm: Model):
    lm += "<body>\n"
    lm += one_or_more(select(options=[_gen_heading(), one_or_more(_gen_para())]))
    lm += "</body>\n"
    return lm
```
接下来是生成完整 HTML 页面的函数。
我们依次添加 HTML 起始标签、生成页面头部、生成页面主体，最后追加 HTML 闭合标签：
```python
@guidance(stateless=True)
def _gen_html(lm: Model):
    lm += "<html>\n"
    lm += _gen_header()
    lm += _gen_body()
    lm += "</html>\n"
    return lm
```
最后，我们提供一个对用户友好的封装函数，支持：
- 设置生成时的采样温度（temperature）
- 从 Model 对象中捕获生成的页面内容
```python
from guidance.library import capture, with_temperature

@guidance(stateless=True)
def make_html(
    lm,
    name: str | None = None,
    *,
    temperature: float = 0.0,
):
    return lm + capture(
        with_temperature(_gen_html(), temperature=temperature),
        name=name,
    )
```
现在，使用该函数来生成一个简单的网页：
```python
lm = phi_lm

with system():
    lm += "You are an expert in HTML"

with user():
    lm += "Create a simple and short web page about your life story."

with assistant():
    lm += make_html(name="html_text", temperature=0.7)
```

在 Jupyter Notebook 中运行并启用交互式小部件时，我们会得到如下输出：

<img src="docs/figures/widget_make_html_20250703.png" alt="Guidance widget showing HTML generation with token fast-forwarding" />

请注意生成结果中不同颜色的高亮显示。
这展示了 Guidance 的另一项核心能力：**词元快进（Token Fast-Forwarding）**。
文法所施加的结构约束通常意味着某些词元（token）是预先确定的。
Guidance 无需让模型去生成这些已确定的词元，而是可以直接将它们插入到生成流中。
这省去了模型的多步前向传播（forward pass），从而大幅降低了 GPU 资源消耗与推理延迟。
例如，在上面的 HTML 生成过程中，Guidance 始终清楚最近一个打开的标签是什么。
如果最近打开的标签是 `<h1>`，那么一旦模型生成了 `</`，Guidance 就能直接补全 `h1>`，而无需模型再执行一次前向传播。

### 生成 JSON

JSON Schema 本质上也是一种上下文无关文法，因此可以通过 Guidance 用来约束大语言模型的输出。
由于这一场景非常普遍，Guidance 为其提供了专门的内置支持。
下面是一个基于 Pydantic 模型的快速示例：
```python
import json
from pydantic import BaseModel, Field

from guidance import json as gen_json

class BloodPressure(BaseModel):
    systolic: int = Field(gt=300, le=400)
    diastolic: int = Field(gt=0, le=20)
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

# 使用 Python 标准库 json 解析
loaded_json = json.loads(lm["bp"])
print(json.dumps(loaded_json, indent=4))

# 使用 Pydantic 验证并格式化
result = BloodPressure.model_validate_json(lm["bp"])
print(result.model_dump_json(indent=8))
```
```
lm['bp']='{"systolic": 301, "diastolic": 15, "location": "arm"}'
{
    "systolic": 301,
    "diastolic": 15,
    "location": "arm"
}
{
        "systolic": 301,
        "diastolic": 15,
        "location": "arm"
}
```
请注意，这里生成的血压数值在真实人类身上几乎不可能出现（模型在训练集中也不会见过这样的人类血压值），但它依然严格服从了我们在字段上定义的数值范围约束。
此外，在生成 JSON 时，由于 Schema 施加了严格的结构约束，大量格式与键名词元同样可以通过词元快进（Fast-Forwarding）机制直接注入，从而显著加速生成。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月17日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
