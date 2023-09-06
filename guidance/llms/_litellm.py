from litellm import acompletion

from ._openai import OpenAI, prompt_to_messages, add_text_to_chat_mode
import litellm

class LiteLLM(OpenAI):
    async def _library_call(self, **kwargs):
        """ Call the LLM APIs using LiteLLM: https://github.com/BerriAI/litellm/"""

        prev_key = litellm.api_key
        prev_org = litellm.organization
        prev_version = litellm.api_version
        prev_base = litellm.api_base

        # set the params of the openai library if we have them
        if self.api_key is not None:
            litellm.api_key = self.api_key
        if self.organization is not None:
            litellm.organization = self.organization
        if self.api_version is not None:
            litellm.api_version = self.api_version
        if self.api_base is not None:
            litellm.api_base = self.api_base

        if self.chat_mode:
            kwargs['messages'] = prompt_to_messages(kwargs['prompt'])
            del kwargs['prompt']
            del kwargs['echo']
            del kwargs['logprobs']
            # print(kwargs)
            out = await acompletion(**kwargs)
            out = add_text_to_chat_mode(out)
        else:
            out = await acompletion(**kwargs)
        
        # restore the params of the openai library
        litellm.api_key = prev_key
        litellm.organization = prev_org
        litellm.api_version = prev_version
        litellm.api_base = prev_base
        return out
