import sys
sys.path.append('./guidance');
import guidance

def main():
        # set the default language model used to execute guidance programs
        # guidance.llm = guidance.llms.OpenAI('gpt-3.5-turbo-FAKE')
    guidance.llm = guidance.llms.Claude('claude-v1-100k')

    # define a guidance program that adapts a proverb
    program = guidance("""Tweak this proverb to apply to model instructions instead.

        {{proverb}}
        - {{book}} {{chapter}}:{{verse}}

        UPDATED
        Where there is no guidance{{gen 'rewrite' stop="\\n-"}}
        - GPT {{gen 'chapter'}}:{{gen 'verse'}}""")

    executed_program = program(
        proverb="Where there is no guidance, a people falls,\nbut in an abundance of counselors there is safety.",
        book="Proverbs",
        chapter=11,
        verse=14
    )

    print(executed_program)

main()
