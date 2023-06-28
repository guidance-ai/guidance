import sys
sys.path.append('./guidance');
import guidance
from decouple import config

OPENAI_API_KEY = config('OPENAI_API_KEY')
def main():
    # set the default language model used to execute guidance programs
    # guidance.llm = guidance.llms.OpenAI('text-davinci-003')
    guidance.llm = guidance.llms.Anthropic('claude-1-100k')
    print('Used LLM: ', guidance.llm);
    # define a guidance program that adapts a proverb
    program = guidance("""\n\nHuman: Tweak this proverb to apply to model instructions instead and generate 3 iterations.

        {{proverb}}
        - {{book}} {{chapter}}:{{verse}}

        UPDATED
        \n\nAssistant: Where there is no guidance{{gen 'rewrite' stop="\\n-"}}
        - GPT {{gen 'chapter'}}:{{gen 'verse'}}""")

    executed_program = program(
        proverb="Where there is no guidance, a people falls,\nbut in an abundance of counselors there is safety.",
        book="Proverbs",
        chapter=11,
        verse=14
    )

    print('executed program: ', executed_program)

main()
