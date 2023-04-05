<div align="center"><img src="docs/figures/guidance_logo_blue.svg" width=300"></div>
<br/>
<b>Guidance</b> goes beyond traditional prompting, templating, and chaining by defining a rich <i>guidance language</i> that expands the API of model language models. It makes it easy to write prompts / programs to control language models with rich output structure.  
Simple output structure like [Chain of Thought](https://arxiv.org/abs/2201.11903) and its many variants (e.g. with [ART](https://arxiv.org/abs/2303.09014),) has been shown to improve LLM performance.  
The advent of more powerful LLMs like [GPT-4](https://arxiv.org/abs/2303.12712) allows for even richer output structure, and `guidance` makes that structure easier and cheaper.

Features:
- [x] Simple, intuitive syntax, using [handlebars](https://handlebarsjs.com/) templating
- [x] Rich output structure with multiple generations, selections, conditionals, tool use, etc
- [x] Playground-like streaming in Jupyter/VSCode Notebooks
- [x] Smart seed-based generation caching
- [x] Support for [OpenAI's Chat models](https://beta.openai.com/docs/guides/chat)
- [x] Easy integration with huggingface models, with [guidance acceleration](for#internal_docs) speedups over standard prompt
- [x] [Token healing](for#link_internal_docs) to optimize guidance boundaries.

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
![anachronism](docs/figures/anachronism.png)

We [compute accuracy](notebooks/anachronism.ipynb) on the validation set, and compare it to using the same two-shot examples above **without** the output structure, as well as to the best reported result [here](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/anachronisms). The results below agree with existing literature, in that even a very simple output structure drastically improves performance, even compared against much larger models.
| Model | Accuracy |
| :---: | :---: |
| [Few-shot learning with guidance examples, no CoT output structure](notebooks/anachronism.ipynb) | 63.04% |
| [PALM (3-shot)](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/anachronisms) | Around 69% |
| [Guidance](notebooks/anachronism.ipynb) | **76.01%** |


## Output structure with OpenAI's Chat models
Full notebook [here](notebooks/chat_topk.ipynb)
```python
import guidance
import re

guidance.llm = guidance.llms.OpenAI("gpt-4", chat_completion=True)

def parse_best(prosandcons, options):
    best = int(re.findall(r'Best=(\d+)', prosandcons)[0])
    return options[best]

create_plan = guidance('''<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
I want to {{goal}}.

{{~! generate potential options ~}}
{{#block hidden=True}}
Can you please generate one option for how to accomplish this?
Please make the option very short, at most one line.
<|im_end|>
<|im_start|>assistant
{{gen 'options' n=5 temperature=1.0 max_tokens=500}}
<|im_end|>
{{/block}}

{{~! generate pros and cons and select the best option ~}}
{{#block hidden=True}}
Can you please comment on the pros and cons of each of the following options, and then pick the best option?
---{{#each options}}
Option {{@index}}: {{this}}{{/each}}
---
Please discuss each option very briefly (one line for pros, one for cons), and end by saying Best=X, where X is the best option.
<|im_end|>
<|im_start|>assistant
{{gen 'prosandcons' temperature=0.0 max_tokens=500}}
<|im_end|>
{{/block}}

{{~! Create a plan }}
Here is my plan:
{{parse_best prosandcons options}}
Please elaborate on this plan, and tell me how to best accomplish it.
<|im_end|>
<|im_start|>assistant
{{gen 'plan' max_tokens=500}}
<|im_end|>''')
out = create_plan(goal='read more books', parse_best=parse_best)
```

This prompt is a bit more complicated, but we are basically going through 3 steps:
1. Generate a few options for how to accomplish the goal. Note that we generate with `n=5`, such that each option is a separate generation (and is not impacted by the other options). We set `temperature=1` to encourage diversity.
2. Generate pros and cons for each option, and select the best one. We set `temperature=0` to encourage the model to be more precise.
3. Generate a plan for the best option, and ask the model to elaborate on it. Notice that steps 1 and 2 were `hidden`, and thus GPT-4 does not see them. This is a simple way to make the model focus on the current step.

Since steps 1 and 2 are hidden, they do not appear on the generated output, but we can print them:
```python
print('\n'.join(['Option %d: %s' % (i, x) for i, x in enumerate(out['options'])]))
```
> Option 0: Set a goal to read for 20 minutes every day before bedtime.  
Option 1: Join a book club for increased motivation and accountability.  
Option 2: Set a daily goal to read for 20 minutes.  
Option 3: Set a daily reminder to read for at least 20 minutes.  
Option 4: Set a daily goal to read at least one chapter or 20 pages.  

```python
print(out['prosandcons'])
``` 

> Option 0:  
> Pros: Establishes a consistent reading routine.  
> Cons: May not be suitable for those with varying schedules.  
> \---  
> Option 1:  
> Pros: Provides social motivation and accountability.  
> Cons: May not align with personal reading preferences.  
> \---  
> Option 2:  
> Pros: Encourages daily reading habit.  
> Cons: Lacks a specific time frame, which may lead to procrastination.  
> \---  
> Option 3:  
> Pros: Acts as a daily reminder to prioritize reading.  
> Cons: May become repetitive and easy to ignore.  
> \---  
> Option 4:  
> Pros: Sets a clear daily reading target.  
> Cons: May be difficult to achieve on busy days or with longer chapters.  
>   
> Best=0  

Here is the final output:
```python
out
```
![chat_reading](docs/figures/chat_reading.png)

## Agents
We can easily build agents that talk to each other or to a user, via the `await` command. For example, here is how we might get GPT-4 to simulate two agents talking to one another:

```python
import guidance
guidance.llm = guidance.llms.OpenAI("gpt-4")

role_simulator = guidance('''<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
You will answer the user as {{role}} in the following conversation. At every step, I will provide you with the user input, as well as a comment reminding you of your instructions. Never talk about the fact that you are an AI, even if the user asks you. Always answer as {{role}}.
{{#if first_question}}You can also start the conversation.{{/if}}<|im_end|>

{{~! The assistant either always starts the conversation }}
<|im_start|>assistant
Ok, I will follow these instructions.
{{#if first_question}}Let me start the conversation now:
{{role}}: {{first_question}}{{/if}}<|im_end|>

{{~! Then the conversation unrolls }}
{{~#geneach 'conversation'}}
<|im_start|>user
User: {{set 'this.input' (await 'input')}}
Comment: Remember, answer as a {{role}}. Start your utterance with {{role}}:<|im_end|>
<|im_start|>assistant
{{gen 'this.response' stop="<|im_end|>" temperature=0 max_tokens=300}}<|im_end|>{{~/geneach}}''')

republican = role_simulator(role='Republican')
democrat = role_simulator(role='Democrat')

first_question = '''What do you think is the best way to stop inflation?'''
republican = republican(input=first_question, first_question=None)
democrat = democrat(input=republican["conversation"][-2]["response"].strip('Republican: '), first_question=first_question)
for i in range(2):
    republican = republican(input=democrat["conversation"][-2]["response"].replace('Democrat: ', ''))
    democrat = democrat(input=republican["conversation"][-2]["response"].replace('Republican: ', ''))

print('Democrat: ' + first_question)
for x in democrat['conversation'][:-1]:
    print('Republican:', x['input'])
    print(x['response'])
    print('')
```
> Democrat: What do you think is the best way to stop inflation?  

> Republican: The best way to stop inflation is by implementing sound fiscal policies, such as reducing government spending, lowering taxes, and promoting economic growth. Additionally, the Federal Reserve should focus on maintaining a stable monetary policy to control inflation.

> Democrat: While I agree that sound fiscal policies are important, as a Democrat, I believe that the government should invest in social programs, education, and infrastructure to promote economic growth and reduce income inequality. Additionally, the Federal Reserve should maintain a stable monetary policy, but also consider the impact of its decisions on employment and overall economic well-being.

> Republican: While investing in social programs, education, and infrastructure is important, we must also prioritize fiscal responsibility and avoid excessive government spending. By lowering taxes and reducing regulations, we can encourage private sector growth and job creation, which will ultimately benefit all Americans. Balancing these investments with responsible spending will help maintain a stable economy and address income inequality without causing inflation. The Federal Reserve should continue to focus on maintaining a stable monetary policy, while also considering the impact of its decisions on employment and overall economic well-being.

> Democrat: I understand the importance of fiscal responsibility, but as a Democrat, I believe that strategic government investments in social programs, education, and infrastructure can lead to long-term economic growth and a more equitable society. We can balance these investments with responsible spending by closing tax loopholes and ensuring that corporations and the wealthy pay their fair share. This approach will help address income inequality and provide opportunities for all Americans, while still maintaining a stable economy. The Federal Reserve should continue to focus on maintaining a stable monetary policy, while also considering the impact of its decisions on employment and overall economic well-being.

> Republican: While strategic government investments can contribute to long-term economic growth, it's crucial to ensure that these investments are made efficiently and do not lead to unsustainable spending. As Republicans, we believe in a limited government role and emphasize the importance of private sector growth to drive economic prosperity. By lowering taxes and reducing regulations, we can create a business-friendly environment that encourages innovation and job creation. Closing tax loopholes and ensuring that everyone pays their fair share is important, but we must also prioritize spending cuts and fiscal responsibility to maintain a stable economy. The Federal Reserve should continue to focus on maintaining a stable monetary policy, while also considering the impact of its decisions on employment and overall economic well-being.

> Democrat: I understand the Republican perspective on limited government and the importance of private sector growth. However, as a Democrat, I believe that the government has a crucial role to play in addressing social and economic issues that the private sector may not prioritize. By investing in social programs, education, and infrastructure, we can create a strong foundation for long-term economic growth and a more equitable society. We can achieve fiscal responsibility by closing tax loopholes, ensuring that everyone pays their fair share, and making smart investments that yield long-term benefits. The Federal Reserve should continue to focus on maintaining a stable monetary policy, while also considering the impact of its decisions on employment and overall economic well-being.


# API reference
All of the examples below are in [this notebook](notebooks/tutorial.ipynb)
## Template syntax
The template syntax is based on [Handlebars](https://handlebarsjs.com/), with a few additions.   
When `guidance` is called, it returns a Program:
```python
prompt = guidance('''What is {{example}}?''')
prompt
```
> What is {{example}}?

The program can be executed by passing in arguments:
```python
prompt(example='truth')
```
> What is truth?

Arguments can be iterables:
```python
people = ['John', 'Mary', 'Bob', 'Alice']
ideas = [{'name': 'truth', 'description': 'the state of being the case'},
         {'name': 'love', 'description': 'a strong feeling of affection'},]
prompt = guidance('''List of people:
{{#each people}}- {{this}}
{{~! This is a comment. The ~ removes adjacent whitespace either before or after a tag, depending on where you place it}}
{{/each~}}
List of ideas:
{{#each ideas}}{{this.name}}: {{this.description}}
{{/each}}''')
prompt(people=people, ideas=ideas)
```
![template_objects](docs/figures/template_objs.png)

Notice the special `~` character after `{{/each}}`.  
This can be added before or after any tag to remove all adjacent whitespace. Notice also the comment syntax: `{{! This is a comment }}`.

You can also include prompts / programs inside other prompts, e.g. here is how you could rewrite the prompt above:
```python
prompt1 = guidance('''List of people:
{{#each people}}- {{this}}
{{/each~}}''')
prompt2 = guidance('''{{>prompt1}}
List of ideas:
{{#each ideas}}{{this.name}}: {{this.description}}
{{/each}}''')
prompt2(prompt1=prompt1, people=people, ideas=ideas)
```

## Generation
### Basic generation
The `gen` tag is used to generate text. You can use whatever arguments are supported by the underlying model.
Executing a prompt calls the generation prompt:
```python
import guidance
# Set the default llm. Could also pass a different one as argument to guidance(), with guidance(llm=...)
guidance.llm = guidance.llms.OpenAI("text-davinci-003")
prompt = guidance('''The best thing about the beach is {{~gen 'best' temperature=0.7 max_tokens=7}}''')
prompt = prompt()
prompt
```
![generation1](docs/figures/generation1.png)  

`guidance` caches all OpenAI generations with the same arguments. If you want to flush the cache, you can call `guidance.llms.OpenAI.cache.clear()`.

### Selecting
You can select from a list of options using the `select` tag:
```python
prompt = guidance('''Is the following sentence offensive? Please answer with a single word, either "Yes", "No", or "Maybe".
Sentence: {{example}}
Answer:{{#select "answer" logprobs='logprobs'}} Yes{{or}} No{{or}} Maybe{{/select}}''')
prompt = prompt(example='I hate tacos')
prompt
```
![select](docs/figures/select.png)
```python
prompt['logprobs']
```
>{' Yes': -1.5689583, ' No': -7.332395, ' Maybe': -0.23746304}

### Sequences of generate / select
A prompt may contain multiple generations or selections, which will be executed in order:
```python
prompt = guidance('''Generate a response to the following email:
{{email}}.
Response:{{gen "response"}}

Is the response above offensive in any way? Please answer with a single word, either "Yes" or "No".
Answer:{{#select "answer" logprobs='logprobs'}} Yes{{or}} No{{/select}}''')
prompt = prompt(email='I hate tacos')
prompt
```
![generate_select](docs/figures/generate_select.png)
```python
prompt['response'], prompt['answer']
```
>(" That's too bad! Tacos are one of my favorite meals.", ' No')

### Hidden generation
You can generate text without displaying it or using it in the subsequent generations using the `hidden` tag, either in a `block` or in a `gen` tag:
```python
prompt = guidance('''{{#block hidden=True}}Generate a response to the following email:
{{email}}.
Response:{{gen "response"}}{{/block}}
I will show you an email and a response, and you will tell me if it's offensive.
Email: {{email}}.
Response: {{response}}
Is the response above offensive in any way? Please answer with a single word, either "Yes" or "No".
Answer:{{#select "answer" logprobs='logprobs'}} Yes{{or}} No{{/select}}''')
prompt = prompt(email='I hate tacos')
prompt
```
![hidden1](docs/figures/hidden1.png)

Notice that nothing inside the hidden block shows up in the output (or was used by the `select`), even though we used the `response` generated variable in the subsequent generation.

### Generate with `n>1`
If you use `n>1`, the variable will contain a list (there is a visualization that lets you navigate the list too):
```python
prompt = guidance('''The best thing about the beach is {{~gen 'best' n=3 temperature=0.7 max_tokens=7}}''')
prompt = prompt()
prompt['best']
```
> [' that it is a great place to',
 ' being able to relax in the sun',
 " that it's a great place to"]

 ## Calling functions
 You can call any python function using generated variables as arguments. The function will be called when the prompt is executed:
 ```python
def aggregate(best):
    return '\n'.join(['- ' + x for x in best])
prompt = guidance('''The best thing about the beach is {{~gen 'best' n=3 temperature=0.7 max_tokens=7 hidden=True}}
{{aggregate best}}''')
prompt = prompt(aggregate=aggregate)
prompt
```
![function](docs/figures/function.png)

## Pausing execution with `await`
An `await` tag will stop program execution until that variable is provided:
```python
prompt = guidance('''Generate a response to the following email:
{{email}}.
Response:{{gen "response"}}
{{await 'instruction'}}
{{gen 'updated_response'}}''', stream=True)
prompt = prompt(email='Hello there')
prompt
```
![await1](docs/figures/await1.png)

Notice how the lest `gen` is not executed because it depends on `instruction`. Let's provide `instruction` now.

```python
prompt = prompt(instruction='Please translate the response above to Portuguese.')
prompt
```
![await2](docs/figures/await2.png)

The program is now executed all the way to the end.

## Notebook functions
Echo, stream. TODO @SCOTT

## Chat