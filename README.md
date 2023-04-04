# Guidance

Guidance makes it easy to write prompts / programs to control language models with rich output structure.  
Simple output structure like [Chain of Thought](https://arxiv.org/abs/2201.11903) and its many variants (e.g. with [ART](https://arxiv.org/abs/2303.09014),) has been shown to improve LLM performance.  
The advent of more powerful LLMs like [GPT-4](https://arxiv.org/abs/2303.12712) allows for even richer output structure, and `guidance` makes that structure easier and cheaper.

# Install

```python
pip install guidance
```

# Quick demos
## Simple output structure
Let's take [a simple task](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/anachronisms) from BigBench, where the goal is to identify whether a given sentence contains an anachronism.  
Here is a simple two-shot prompt for it, with a human-crafted chain-of-thought sequence:
```python
import guidance
guidance.llm = guidance.llms.OpenAI("text-davinci-003") 
instruction = 'Given a sentence tell me whether it contains an anachronism (i.e. whether it could have happened or not based on the time periods associated with the entities).'
examples = [
    {'input': 'I wrote about shakespeare',
    'entities': [{'entity': 'I', 'time': 'present'}, {'entity': 'Shakespeare', 'time': '16th century'}],
    'reasoning': 'I can write about Shakespeare because he lived in the past with respect to me.',
    'answer': 'No'},
    {'input': 'Shakespeare wrote about me',
    'entities': [{'entity': 'Shakespeare', 'time': '16th century'}, {'entity': 'I', 'time': 'present'}],
    'reasoning': 'Shakespeare cannot have written about me, because he died before I was born',
    'answer': 'Yes'}
    ]
structure_prompt = guidance(
'''{{instruction}}
----
{{~! Few shot examples here ~}}
{{~#each examples}}
Sentence: {{this.input}}
Entities and dates:{{#each this.entities}}
{{this.entity}}: {{this.time}}{{/each}}
Reasoning: {{this.reasoning}}
Anachronism: {{this.answer}}
---
{{~/each}}
{{~! Input example here}}
Sentence: {{input}}
Entities and dates:
{{gen "entities"}}
Reasoning:{{gen "reasoning"}}
Anachronism:{{#select "answer"}} Yes{{or}} No{{/select}}''')
structure_prompt(examples=examples, input='The T-rex bit my dog', instruction=instruction)
```

<pre style='margin: 0px; padding: 0px; padding-left: 8px; margin-left: -8px; border-radius: 0px; border-left: 1px solid rgba(127, 127, 127, 0.2); white-space: pre-wrap; font-family: ColfaxAI, Arial; font-size: 15px; line-height: 23px;'><span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;instruction&amp;#125;&amp;#125;'>Given a sentence tell me whether it contains an anachronism (i.e. whether it could have happened or not based on the time periods associated with the entities).</span>
----<span style='opacity: 1.0; display: inline; background-color: rgba(0, 138.56128016, 250.76166089, 0.25);' title='&amp;#123;&amp;#123;~#each examples&amp;#125;&amp;#125;
Sentence: &amp;#123;&amp;#123;this.input&amp;#125;&amp;#125;
Entities and dates:&amp;#123;&amp;#123;#each this.entities&amp;#125;&amp;#125;
&amp;#123;&amp;#123;this.entity&amp;#125;&amp;#125;: &amp;#123;&amp;#123;this.time&amp;#125;&amp;#125;&amp;#123;&amp;#123;/each&amp;#125;&amp;#125;
Reasoning: &amp;#123;&amp;#123;this.reasoning&amp;#125;&amp;#125;
Anachronism: &amp;#123;&amp;#123;this.answer&amp;#125;&amp;#125;
---
&amp;#123;&amp;#123;~/each&amp;#125;&amp;#125;'>
Sentence: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.input&amp;#125;&amp;#125;'>I wrote about shakespeare</span>
Entities and dates:<span style='opacity: 1.0; display: inline; background-color: rgba(0, 138.56128016, 250.76166089, 0.25);' title='&amp;#123;&amp;#123;#each this.entities&amp;#125;&amp;#125;
&amp;#123;&amp;#123;this.entity&amp;#125;&amp;#125;: &amp;#123;&amp;#123;this.time&amp;#125;&amp;#125;&amp;#123;&amp;#123;/each&amp;#125;&amp;#125;'>
<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.entity&amp;#125;&amp;#125;'>I</span>: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.time&amp;#125;&amp;#125;'>present</span>
<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.entity&amp;#125;&amp;#125;'>Shakespeare</span>: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.time&amp;#125;&amp;#125;'>16th century</span></span>
Reasoning: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.reasoning&amp;#125;&amp;#125;'>I can write about Shakespeare because he lived in the past with respect to me.</span>
Anachronism: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.answer&amp;#125;&amp;#125;'>No</span>
---
Sentence: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.input&amp;#125;&amp;#125;'>Shakespeare wrote about me</span>
Entities and dates:<span style='opacity: 1.0; display: inline; background-color: rgba(0, 138.56128016, 250.76166089, 0.25);' title='&amp;#123;&amp;#123;#each this.entities&amp;#125;&amp;#125;
&amp;#123;&amp;#123;this.entity&amp;#125;&amp;#125;: &amp;#123;&amp;#123;this.time&amp;#125;&amp;#125;&amp;#123;&amp;#123;/each&amp;#125;&amp;#125;'>
<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.entity&amp;#125;&amp;#125;'>Shakespeare</span>: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.time&amp;#125;&amp;#125;'>16th century</span>
<span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.entity&amp;#125;&amp;#125;'>I</span>: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.time&amp;#125;&amp;#125;'>present</span></span>
Reasoning: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.reasoning&amp;#125;&amp;#125;'>Shakespeare cannot have written about me, because he died before I was born</span>
Anachronism: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;this.answer&amp;#125;&amp;#125;'>Yes</span>
---</span>
Sentence: <span style='background-color: rgba(0, 138.56128016, 250.76166089, 0.25); display: inline;' title='&amp;#123;&amp;#123;input&amp;#125;&amp;#125;'>The T-rex bit my dog</span>
Entities and dates:
<span style='background-color: rgba(0, 165, 0, 0.25); opacity: 1.0; display: inline;' title='&amp;#123;&amp;#123;gen &quot;entities&quot;&amp;#125;&amp;#125;'>T-rex: 65 million years ago
My dog: present</span>
Reasoning:<span style='background-color: rgba(0, 165, 0, 0.25); opacity: 1.0; display: inline;' title='&amp;#123;&amp;#123;gen &quot;reasoning&quot;&amp;#125;&amp;#125;'> The T-rex lived millions of years before my dog, so it cannot have bitten my dog.</span>
Anachronism:<span style='background-color: rgba(0, 165, 0, 0.25); opacity: 1.0; display: inline;' title='&amp;#123;&amp;#123;#select &quot;answer&quot;&amp;#125;&amp;#125; Yes&amp;#123;&amp;#123;or&amp;#125;&amp;#125; No&amp;#123;&amp;#123;/select&amp;#125;&amp;#125;'> Yes</span></pre>


We [compute accuracy](notebooks/anachronism.ipynb) on the validation set, and compare it to using the same two-shot examples above **without** the output structure, as well as to the best reported result [here](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/anachronisms). The results below agree with existing literature, in that even a very simple output structure drastically improves performance, even compared against much larger models.
| Model | Accuracy |
| :---: | :---: |
| [Few-shot learning with guidance examples, no CoT output structure](notebooks/anachronism.ipynb) | 63.04% |
| [PALM (3-shot)](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/anachronisms) | Around 69% |
| [Guidance](notebooks/anachronism.ipynb) | **76.01%** |


## Output structure with OpenAI's Chat models

