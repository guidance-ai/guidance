import vertexai.language_models as palm
import os
import time
import requests
import aiohttp
import copy
import time
import asyncio
import types
import collections
import json
import re
import regex
import os
import string

from ._llm import LLM, LLMSession, SyncSession


class MalformedPromptException(Exception):
    pass


import pyparsing as pp

role_start_tag = pp.Suppress(pp.Optional(pp.White()) + pp.Literal("<|im_start|>"))
role_start_name = pp.Word(pp.alphanums + "_")("role_name")
role_kwargs = pp.Suppress(pp.Optional(" ")) + pp.Dict(
    pp.Group(pp.Word(pp.alphanums + "_") + pp.Suppress("=") + pp.QuotedString('"'))
)("kwargs")
role_start = (
    role_start_tag + role_start_name + pp.Optional(role_kwargs) + pp.Suppress("\n")
).leave_whitespace()
role_end = pp.Suppress(pp.Literal("<|im_end|>"))
role_content = pp.Combine(
    pp.ZeroOrMore(pp.CharsNotIn("<") | pp.Literal("<") + ~pp.FollowedBy("|im_end|>"))
)("role_content")
role_group = pp.Group(role_start + role_content + role_end)(
    "role_group"
).leave_whitespace()
partial_role_group = pp.Group(role_start + role_content)(
    "role_group"
).leave_whitespace()
roles_grammar = (
    pp.ZeroOrMore(role_group) + pp.Optional(partial_role_group) + pp.StringEnd()
)

input_start_tag = pp.Suppress(pp.Optional(pp.White()) + pp.Literal("<|input_start|>"))
input_start = (input_start_tag + pp.Suppress("\n")).leave_whitespace()
input_end = pp.Suppress(pp.Literal("<|input_end|>"))
input_content = pp.Combine(
    pp.ZeroOrMore(
        pp.CharsNotIn("<")
        | pp.Literal("<") + ~pp.FollowedBy("|input_end|>") + ~pp.FollowedBy("|im_end|>")
    )
)("input_content")
input_group = pp.Group(input_start + input_content + input_end)(
    "input_group"
).leave_whitespace()

output_start_tag = pp.Suppress(pp.Optional(pp.White()) + pp.Literal("<|output_start|>"))
output_start = (output_start_tag + pp.Suppress("\n")).leave_whitespace()
output_end = pp.Suppress(pp.Literal("<|output_end|>"))
output_content = pp.Combine(
    pp.ZeroOrMore(
        pp.CharsNotIn("<")
        | pp.Literal("<")
        + ~pp.FollowedBy("|output_end|>")
        + ~pp.FollowedBy("|im_end|>")
    )
)("output_content")
output_group = pp.Group(output_start + output_content + output_end)(
    "output_group"
).leave_whitespace()
input_output_grammar = input_group + output_group + pp.StringEnd()


def prompt_to_messages(prompt):
    messages = []

    assert prompt.endswith(
        "<|im_start|>assistant\n"
    ), "When calling PaLM chat models you must generate only directly inside the assistant role! The PaLM API does not currently support partial assistant prompting."

    parsed_prompt = roles_grammar.parse_string(prompt)

    # pattern = r'<\|im_start\|>([^\n]+)\n(.*?)(?=<\|im_end\|>|$)'
    # matches = re.findall(pattern, prompt, re.DOTALL)

    # if not matches:
    #     return [{'role': 'user', 'content': prompt}]

    for role in parsed_prompt:
        if len(role["role_content"]) > 0:  # only add non-empty messages
            message = {"role": role["role_name"], "content": role["role_content"]}
            if "kwargs" in role:
                for k, v in role["kwargs"].items():
                    message[k] = v
            messages.append(message)

    return messages


async def convert_to_guidance_names_generator(chat_mode):
    for resp in chat_mode:
        if resp.is_blocked:
            # raise Exception("The model blocked the request")
            # on streams, google allows the last one to have blocked=True
            break
        yield {"choices": [{"text": resp.text}]}


async def convert_to_guidance_names_agenerator(chat_mode):
    async for resp in chat_mode:
        safety_attributes = resp["outputs"][0]["structVal"]["safetyAttributes"]
        if "listVal" in safety_attributes:
            safety_attributes = safety_attributes["listVal"][0]
        if safety_attributes["structVal"]["blocked"]["boolVal"][0]:
            # raise Exception("The model blocked the request")
            # on streams, google allows the last one to have blocked=True
            break
        output = resp["outputs"][0]["structVal"]
        if "content" in output:
            yield {"choices": [{"text": output["content"]["stringVal"][0]}]}
        else:
            output = output["candidates"]
            yield {
                "choices": [
                    {"text": v["structVal"]["content"]["stringVal"][0]}
                    for v in output["listVal"]
                ]
            }


def convert_to_guidance_names(chat_mode):
    if isinstance(chat_mode, (palm.TextGenerationResponse)):
        chat_mode = {"choices": [{"text": chat_mode.text}]}
    elif isinstance(chat_mode, (types.AsyncGeneratorType)):
        return convert_to_guidance_names_agenerator(chat_mode)
    elif isinstance(chat_mode, (types.GeneratorType)):
        return convert_to_guidance_names_generator(chat_mode)
    else:
        if "predictions" in chat_mode:
            chat_mode = chat_mode["predictions"][0]
            if "candidates" in chat_mode:
                for c in chat_mode["candidates"]:
                    c["text"] = c.pop("content")
                chat_mode["choices"] = chat_mode.pop("candidates")
            else:
                chat_mode["choices"] = [{"text": chat_mode["content"]}]
        else:
            return {
                "choices": [
                    {
                        "text": chat_mode[0]["outputs"][0]["structVal"]["content"][
                            "stringVal"
                        ][0]
                    }
                ]
            }
    return chat_mode


def aggregate_messages(messages):
    """Aggregates all system messages to form the context and aggregates all examples in the prompt.
    Returns the context string, the examples and the message list.
    """
    context = ""
    examples = []
    aggregated_messages = []
    for message in messages:
        if message["role"] == "system" or message["role"] == "context":
            context += message["content"]
        elif message["role"] == "example":
            parsed_example = input_output_grammar.parse_string(message["content"])
            examples.append(
                {
                    "input": {"content": parsed_example[0]["input_content"]},
                    "output": {"content": parsed_example[1]["output_content"]},
                }
            )
        else:
            aggregated_messages.append(
                {"author": message["role"], "content": message["content"]}
            )
    return context, examples, aggregated_messages


class PaLM(LLM):
    llm_name: str = "palm"

    def __init__(
        self,
        model=None,
        caching=True,
        max_retries=5,
        max_calls_per_min=60,
        api_key=None,
        api_type=None,
        api_base=None,
        api_version=None,
        temperature=0.0,
        chat_mode="auto",
        project_id=None,
        rest_call=False,
        allowed_special_tokens={"<calc>", "</calc>"},
        endpoint="https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/{model}",
    ):
        super().__init__()

        if endpoint is not None:
            if api_base is None:
                api_base = endpoint

        ##########################
        ### FINDING MODEL NAME ###
        ##########################
        if model is None:
            model = os.environ.get("PALM_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser("~/.palm_model"), "r") as file:
                    model = file.read().replace("\n", "")
            except:
                pass

        ##########################
        ### FINDING CHAT MODE  ###
        ##########################
        if chat_mode == "auto":
            # parse to determin if the model need to use the chat completion API
            chat_model_pattern = r"chat-bison|chat-bison-32k"
            if re.match(chat_model_pattern, model):
                chat_mode = True
            else:
                chat_mode = False

        ##########################
        ###  FINDING API KEY   ###
        ##########################
        if api_key is None:  # get from environment variable
            api_key = os.environ.get("GOOGLE_API_KEY", getattr(palm, "api_key", None))
        if (
            api_key is not None
            and not api_key.startswith("sk-")
            and os.path.exists(os.path.expanduser(api_key))
        ):  # get from file
            with open(os.path.expanduser(api_key), "r") as file:
                api_key = file.read().replace("\n", "")
        if api_key is None:  # get from default file location
            try:
                with open(os.path.expanduser("~/.google_api_key"), "r") as file:
                    api_key = file.read().replace("\n", "")
            except:
                pass

        ##################################
        ###  FINDING GCP CREDENTIALS   ###
        ##################################
        gcp_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", None)

        ##########################
        ### FINDING PROJECT ID ###
        ##########################
        if project_id is None:
            project_id = os.environ.get("GOOGLE_PROJECT_ID", None)

        ##########################
        ###  FINDING ENDPOINT  ###
        ##########################
        if api_base is None:
            api_base = os.environ.get("PALM_API_BASE", None) or os.environ.get(
                "PALM_ENDPOINT", None
            )  # ENDPOINT is deprecated
            if api_base is not None:
                with api_base.rfind(":") as rcolon_index:
                    if rcolon_index != -1 and (
                        api_base[rcolon_index + 1 :] == "serverStreamingPredict"
                        or api_base[rcolon_index + 1 :] == "predict"
                    ):
                        api_base = api_base[:rcolon_index]

        # TODO: TOKENIZER?

        self.chat_mode = chat_mode

        self.allowed_special_tokens = allowed_special_tokens
        self.model_name = model
        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        if isinstance(api_key, str):
            api_key = api_key.replace("Bearer ", "")
        self.api_key = api_key
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.project_id = project_id
        self.rest_call = rest_call
        self.endpoint = endpoint
        self.gcp_credentials = gcp_credentials

        if not self.rest_call:
            self.caller = self._library_call
            if self.chat_mode:
                self.chat_model = palm.ChatModel.from_pretrained(self.model_name)
            else:
                self.generation_model = palm.TextGenerationModel.from_pretrained(
                    self.model_name
                )
        else:
            self.caller = self._rest_call
            self._rest_headers = {"Content-Type": "application/json"}

    def session(self, asynchronous=False):
        if asynchronous:
            return PALMSession(self)
        else:
            return SyncSession(PALMSession(self))

    def role_start(self, role_name, **kwargs):
        assert self.chat_mode, "role_start() can only be used in chat mode"
        return (
            "<|im_start|>"
            + role_name
            + "".join([f' {k}="{v}"' for k, v in kwargs.items()])
            + "\n"
        )

    def role_end(self, role=None):
        assert self.chat_mode, "role_end() can only be used in chat mode"
        return "<|im_end|>"

    def input_start(self, **kwargs):
        assert self.chat_mode, "input_start() can only be used in chat mode"
        return "<|input_start|>" + "\n"

    def input_end(self):
        assert self.chat_mode, "input_end() can only be used in chat mode"
        return "<|input_end|>"

    def output_start(self, **kwargs):
        assert self.chat_mode, "output_start() can only be used in chat mode"
        return "<|output_start|>" + "\n"

    def output_end(self):
        assert self.chat_mode, "output_end() can only be used in chat mode"
        return "<|output_end|>"

    @classmethod
    async def stream_then_save(cls, gen, key, stop_regex, n):
        list_out = []
        cached_out = None

        # init stop_regex variables
        if stop_regex is not None:
            if isinstance(stop_regex, str):
                stop_patterns = [regex.compile(stop_regex)]
            else:
                stop_patterns = [regex.compile(pattern) for pattern in stop_regex]

            current_strings = ["" for _ in range(n)]
            # last_out_pos = ["" for _ in range(n)]

        # iterate through the stream
        all_done = False
        async for curr_out in gen:
            # if we have a cached output, extend it with the current output
            if cached_out is not None:
                out = merge_stream_chunks(cached_out, curr_out)
            else:
                out = curr_out

            # check if we have stop_regex matches
            found_partial = False
            if stop_regex is not None:
                # keep track of the generated text so far
                for i, choice in enumerate(curr_out["choices"]):
                    current_strings[i] += choice["text"]

                # check if all of the strings match a stop string (and hence we can stop the batch inference)
                all_done = True
                for i in range(len(current_strings)):
                    found = False
                    for s in stop_patterns:
                        if s.search(current_strings[i]):
                            found = True
                    if not found:
                        all_done = False
                        break

                # find where trim off the stop regex matches if needed (and look for partial matches)
                stop_pos = [1e10 for _ in range(n)]
                stop_text = [None for _ in range(n)]
                for i in range(len(current_strings)):
                    for s in stop_patterns:
                        m = s.search(current_strings[i], partial=True)
                        if m:
                            span = m.span()
                            if span[1] > span[0]:
                                if (
                                    m.partial
                                ):  # we might be starting a stop sequence, so we can't emit anything yet
                                    found_partial = True
                                    break
                                else:
                                    stop_text[i] = current_strings[i][span[0] : span[1]]
                                    stop_pos[i] = min(span[0], stop_pos[i])
                    if stop_pos != 1e10:
                        stop_pos[i] = stop_pos[i] - len(
                            current_strings[i]
                        )  # convert to relative position from the end

            # if we might be starting a stop sequence, we need to cache the output and continue to wait and see
            if found_partial:
                cached_out = out
                continue

            # if we get here, we are not starting a stop sequence, so we can emit the output
            else:
                cached_out = None

                if stop_regex is not None:
                    for i in range(len(out["choices"])):
                        if stop_pos[i] < len(out["choices"][i]["text"]):
                            out["choices"][i]["text"] = out["choices"][i]["text"][
                                : stop_pos[i]
                            ]
                            out["choices"][i]["stop_text"] = stop_text[i]
                            out["choices"][i]["finish_reason"] = "stop"

                list_out.append(out)
                yield out
                if all_done:
                    gen.aclose()
                    break

        # if we have a cached output, emit it
        if cached_out is not None:
            list_out.append(cached_out)
            yield out

        cls.cache[key] = list_out

    def _stream_completion(self):
        pass

    def end_of_text(self):
        return "<|endoftext|>"

    # Define a function to add a call to the deque
    def add_call(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Append the timestamp to the right of the deque
        self.call_history.append(now)

    # Define a function to count the calls in the last 60 seconds
    def count_calls(self):
        # Get the current timestamp in seconds
        now = time.time()
        # Remove the timestamps that are older than 60 seconds from the left of the deque
        while self.call_history and self.call_history[0] < now - 60:
            self.call_history.popleft()
        # Return the length of the deque as the number of calls
        return len(self.call_history)

    async def _library_call(self, **kwargs):
        """Call the PaLM API using the python package."""

        # candidate count is not used on library calls
        del kwargs["function_call"]
        del kwargs["candidate_count"]
        kwargs["temperature"] = float(kwargs["temperature"])
        stream = kwargs.pop("stream", False)
        kwargs["stop_sequences"] = kwargs.get("stop_sequences", [])
        if type(kwargs["stop_sequences"]) == str:
            kwargs["stop_sequences"] = [kwargs["stop_sequences"]]
        if self.chat_mode:
            context, examples, messages = aggregate_messages(
                prompt_to_messages(kwargs.pop("prompt"))
            )
            messages = [
                palm.ChatMessage(author=x["author"], content=x["content"])
                for x in messages
            ]
            examples = [
                palm.InputOutputTextPair(
                    input_text=x["input"]["content"], output_text=x["output"]["content"]
                )
                for x in examples
            ]
            kwargs["context"] = context
            kwargs["message_history"] = messages[:-1]
            kwargs["examples"] = examples
            chat = self.chat_model.start_chat(**kwargs)
            if len(messages) == 0:
                messages.append(palm.ChatMessage(author="user", content="\t"))
            if stream:
                out = chat.send_message_streaming(messages[-1].content)
            else:
                out = chat.send_message(messages[-1].content)
        else:
            if stream:
                kwargs.pop("stop_sequences")
                out = self.generation_model.predict_streaming(**kwargs)
            else:
                out = self.generation_model.predict(**kwargs)
        return convert_to_guidance_names(out)

    async def _rest_call(self, **kwargs):
        """Call the PaLM API using the REST API."""

        url = self.api_base
        # Define the request headers
        headers = copy.copy(self._rest_headers)

        assert (
            self.api_key is not None and self.project_id is not None
        ), "You must provide a PaLM API key and project ID to use the PaLM LLM via rest calls. Either pass it in the constructor, set the GOOGLE_API_KEY and GOOGLE_PROJECT_ID environment variables, or create the files ~/.google_api_key and ~/.google_project_id with your credentials."

        stream = kwargs.pop("stream", False)
        if stream:
            url += ":serverStreamingPredict"
        else:
            url += ":predict"

        headers["Authorization"] = f"Bearer {self.api_key}"

        stop_sequences = kwargs.get("stop_sequences", [])
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        # Define the request data
        if stream:
            data = {
                "inputs": [
                    {
                        "struct_val": {
                            "prompt": {
                                "string_val": kwargs["prompt"],
                            }
                        }
                    },
                ],
                "parameters": {
                    "struct_val": {
                        "maxOutputTokens": {
                            "int_val": kwargs.get("max_output_tokens", 0)
                        },
                        "temperature": {
                            "float_val": float(kwargs.get("temperature", 0.0))
                        },
                        "topP": {"float_val": float(kwargs.get("top_p", 0.95))},
                        "topK": {"int_val": kwargs.get("top_k", 40)},
                    }
                },
            }
        else:
            data = {
                "instances": [
                    {
                        "prompt": kwargs["prompt"],
                    },
                ],
                "parameters": {
                    "maxOutputTokens": kwargs.get("max_output_tokens", 0),
                    "temperature": kwargs.get("temperature", 0.0),
                    "topP": kwargs.get("top_p", 0.95),
                    "topK": kwargs.get("top_k", 40),
                    "candidateCount": kwargs.get("candidate_count", 1),
                    "stopSequences": stop_sequences,
                },
            }

        # if stream and not self.chat_mode:
        #    data["parameters"].pop("stop_sequences")

        if self.chat_mode:
            context, examples, messages = aggregate_messages(
                prompt_to_messages(kwargs["prompt"])
            )
            if stream:
                data["inputs"] = [
                    {
                        "structVal": {
                            "messages": {
                                "listVal": [
                                    {
                                        "structVal": {
                                            "content": {"string_val": m["content"]},
                                            "author": {"string_val": m["author"]},
                                        }
                                    }
                                    for m in messages
                                ]
                            }
                        }
                    }
                ]
            else:
                data["instances"] = [
                    {
                        "context": context,
                        "messages": messages,
                        "examples": examples,
                    }
                ]

        # Send a POST request and get the response
        # An exception for timeout is raised if the server has not issued a response for 10 seconds
        try:
            if len(list(string.Formatter().parse(url))) > 0:
                url = url.format(project_id=self.project_id, model=self.model_name)
            if stream:
                session = aiohttp.ClientSession()
                response = await session.post(
                    url, json=data, headers=headers, timeout=60
                )
                status = response.status
            else:
                response = requests.post(url, headers=headers, json=data, timeout=60)
                status = response.status_code
                text = response.text
            if status != 200:
                if stream:
                    text = await response.text()
                raise Exception("Response is not 200: " + text)
            if stream:
                response = self._rest_stream_handler(response, session)
            else:
                response = response.json()
        except requests.Timeout:
            raise Exception("Request timed out.")
        except requests.ConnectionError:
            raise Exception("Connection error occurred.")

        response = convert_to_guidance_names(response)

        return response

    async def _close_response_and_session(self, response, session):
        await response.release()
        await session.close()

    async def _rest_stream_handler(self, response, session):
        # async for line in response.iter_lines():
        lines = []
        async for line in response.content:
            text = line.decode("utf-8")
            if text.startswith('  "outputs": [\n'):
                if len(lines) > 1:
                    yield (json.loads("".join(lines[:-2])))
                lines = ["{\n"]
            lines.append(text)
        yield (json.loads("".join(lines[:-1])))
        await self._close_response_and_session(response, session)

    def encode(self, string):
        # note that is_fragment is not used used for this tokenizer
        return self._tokenizer.encode(
            string, allowed_special=self.allowed_special_tokens
        )

    def decode(self, tokens):
        return self._tokenizer.decode(tokens)


def merge_stream_chunks(first_chunk, second_chunk):
    """This merges two stream responses together."""

    out = copy.deepcopy(first_chunk)

    # merge the choices
    for i in range(len(out["choices"])):
        out_choice = out["choices"][i]
        second_choice = second_chunk["choices"][i]
        out_choice["text"] += second_choice["text"]
        if "index" in second_choice:
            out_choice["index"] = second_choice["index"]
        if "finish_reason" in second_choice:
            out_choice["finish_reason"] = second_choice["finish_reason"]
        if out_choice.get("logprobs", None) is not None:
            out_choice["logprobs"]["token_logprobs"] += second_choice["logprobs"][
                "token_logprobs"
            ]
            out_choice["logprobs"]["top_logprobs"] += second_choice["logprobs"][
                "top_logprobs"
            ]
            out_choice["logprobs"]["text_offset"] = second_choice["logprobs"][
                "text_offset"
            ]

    return out


class RegexStopChecker:
    def __init__(self, stop_pattern, decode, prefix_length):
        if isinstance(stop_pattern, str):
            self.stop_patterns = [regex.compile(stop_pattern)]
        else:
            self.stop_patterns = [regex.compile(pattern) for pattern in stop_pattern]
        self.prefix_length = prefix_length
        self.decode = decode
        self.current_strings = None
        self.current_length = 0

    def __call__(self, input_ids, scores, **kwargs):
        # extend our current strings
        if self.current_strings is None:
            self.current_strings = ["" for _ in range(len(input_ids))]
        for i in range(len(self.current_strings)):
            self.current_strings[i] += self.decode(input_ids[i][self.current_length :])

        # trim off the prefix string so we don't look for stop matches in the prompt
        if self.current_length == 0:
            for i in range(len(self.current_strings)):
                self.current_strings[i] = self.current_strings[i][self.prefix_length :]

        self.current_length = len(input_ids[0])

        # check if all of the strings match a stop string (and hence we can stop the batch inference)
        all_done = True
        for i in range(len(self.current_strings)):
            found = False
            for s in self.stop_patterns:
                if s.search(self.current_strings[i]):
                    found = True
            if not found:
                all_done = False
                break

        return all_done


# Define a deque to store the timestamps of the calls
class PALMSession(LLMSession):
    async def __call__(
        self,
        prompt,
        stop=[],
        stop_regex=None,
        temperature=None,
        n=1,
        max_tokens=1000,
        logprobs=None,
        top_p=1.0,
        echo=False,
        logit_bias=None,
        token_healing=None,
        pattern=None,
        stream=None,
        cache_seed=0,
        caching=None,
        **completion_kwargs,
    ):
        """Generate a completion of the given prompt."""

        # we need to stream in order to support stop_regex
        if stream is None:
            stream = stop_regex is not None

        assert (
            stop_regex is None or stream
        ), "We can only support stop_regex for the PaLM API when stream=True!"
        assert (
            stop_regex is None or n == 1
        ), "We don't yet support stop_regex combined with n > 1 with the PaLM API!"

        assert (
            token_healing is None or token_healing is False
        ), "The PaLM API does not yet support token healing! Please either switch to an endpoint that does, or don't use the `token_healing` argument to `gen`."

        # set defaults
        if temperature is None:
            temperature = self.llm.temperature

        # get the arguments as dictionary for cache key generation
        args = locals().copy()

        assert (
            not pattern
        ), "The PaLM API does not support Guidance pattern controls! Please either switch to an endpoint that does, or don't use the `pattern` argument to `gen`."

        # define the key for the cache
        cache_params = self._cache_params(args)
        llm_cache = self.llm.cache
        key = llm_cache.create_key(self.llm.llm_name, **cache_params)

        # allow streaming to use non-streaming cache (the reverse is not true)
        if key not in llm_cache and stream:
            cache_params["stream"] = False
            key1 = llm_cache.create_key(self.llm.llm_name, **cache_params)
            if key1 in llm_cache:
                key = key1

        # check the cache
        if (
            key not in llm_cache
            or caching is False
            or (caching is not True and not self.llm.caching)
        ):
            # ensure we don't exceed the rate limit
            while self.llm.count_calls() > self.llm.max_calls_per_min:
                await asyncio.sleep(1)

            fail_count = 0
            if stop is None:
                stop = []
            while True:
                try_again = False
                try:
                    self.llm.add_call()
                    call_args = {
                        "prompt": prompt,
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "candidate_count": n,
                        "stop_sequences": stop,
                        "stream": stream,
                        **completion_kwargs,
                    }

                    out = await self.llm.caller(**call_args)

                except ValueError:
                    await asyncio.sleep(1)
                    try_again = True
                    fail_count += 1

                if not try_again:
                    break

                if fail_count > self.llm.max_retries:
                    raise Exception(
                        f"Too many (more than {self.llm.max_retries}) PaLM API errors in a row!"
                    )

            if stream:
                return self.llm.stream_then_save(out, key, stop_regex, n)
            else:
                llm_cache[key] = out

        # wrap as a list if needed
        if stream:
            if isinstance(llm_cache[key], list):
                return llm_cache[key]
            return [llm_cache[key]]

        return llm_cache[key]
