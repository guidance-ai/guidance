# mypy: ignore-errors
# NOTE(nopdive): This module has multiple imports from
#   dependencies without stubs / markers. Ignoring mypy for now.
"""Backend powerlift integration for benchmarking."""

from typing import Generator, List, Optional, Tuple, Union
import pathlib
from pathlib import Path


def retrieve_langchain(
    cache_dir: Optional[Union[str, Path]] = None,
) -> Generator[object, None, None]:
    """Retrieves LangChain datasets appropriate for guidance benchmarking. Requires env `LANGCHAIN_API_KEY` to be set on first call.

    Args:
        cache_dir (Optional[Union[str, Path]], optional): Directory to store downloaded datasets. Defaults to None.

    Yields:
        Generator[object, None, None]: DataFrameDataset required for powerlift.
    """

    import pandas as pd
    from langchain_benchmarks import registry
    from langsmith.client import Client
    from langchain_benchmarks import clone_public_dataset
    from powerlift.bench.store import update_cache, retrieve_cache, DataFrameDataset
    import os

    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir, "langchain")

    name = "chat_extract"
    inputs_name = f"{name}.inputs.parquet"
    outputs_name = f"{name}.outputs.parquet"
    meta_name = f"{name}.meta.json"
    cached = retrieve_cache(cache_dir, [inputs_name, outputs_name, meta_name])
    if cached is None:
        if os.getenv("LANGCHAIN_API_KEY") is None:
            raise ValueError("Env variable LANGCHAIN_API_KEY is not set.")

        task = registry["Chat Extraction"]
        clone_public_dataset(task.dataset_id, dataset_name=task.name)
        client = Client()
        dataset = client.read_dataset(dataset_name=task.name)
        examples = list(client.list_examples(dataset_id=dataset.id, as_of=None))

        system_prompt = (
            "You are a data extraction bot tasked with "
            + "extracting and inferring information from dialogues and generating tickets. Always respond "
            + f"only with json based on the following JSON schema:\n{task.schema.schema_json()}"
        )

        user_prompts = []
        for example in examples:
            dialogue = (
                f"<question>\n{example.inputs['question']}\n</question>\n"
                + f"<assistant-response>\n{example.inputs['answer']}\n</assistant-response>"
            )
            user_prompt = (
                "Generate a ticket from the following question-response pair:\n"
                + f"<Dialogue>\n{dialogue}\n</Dialogue>\n"
                + "Remember, respond directly with this format:\n"
                + f'{{"{task.schema.schema()["title"]}": ...}}\n'
                + "RESPOND ONLY IN JSON THEN STOP."
            )
            user_prompts.append(user_prompt)

        inputs = pd.DataFrame.from_records(
            [
                {
                    "input": example.inputs,
                    "schema": task.schema.schema_json(),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompts[i],
                }
                for i, example in enumerate(examples)
            ]
        )

        outputs = pd.DataFrame.from_records(
            [{"output": example.outputs} for example in examples]
        )
        meta = {
            "name": name,
            "problem": "guidance/struct_decode",
            "source": "langchain",
            "inputs_categorical_mask": [dt.kind == "O" for dt in inputs.dtypes],
            "outputs_categorical_mask": [dt.kind == "O" for dt in outputs.dtypes],
            "inputs_feature_names": list(inputs.columns),
            "outputs_feature_names": list(outputs.columns),
        }
        guidance_dataset = DataFrameDataset(inputs, outputs, meta)
    else:
        guidance_dataset = DataFrameDataset.deserialize(*cached)

    if cache_dir is not None:
        serialized = DataFrameDataset.serialize(guidance_dataset)
        update_cache(cache_dir, [inputs_name, outputs_name, meta_name], serialized)

    yield guidance_dataset


def langchain_chat_extract_filter_template(models: List[str], task):
    """Template that filters tasks in benchmark down to langchain chat extract.

    Needs to be used as a partial function with only task as an argument for powerlift.

    Args:
        models (List[str]): Model names to run.
        task: Powerlift task.
    """

    if task.problem == "guidance/struct_decode":
        return models
    return []


def langchain_chat_extract_runner(trial):
    """Runs a single trial for langchain chat extract

    Args:
        trial: Powerlift trial.
    """
    from guidance import models, gen, guidance, select, zero_or_more
    from time import time
    import pandas as pd
    import json
    import json_stream
    import io
    from huggingface_hub import hf_hub_download

    if trial.task.name == "chat_extract":
        inputs, outputs, meta = trial.task.data(["inputs", "outputs", "meta"])
        merged_df = pd.concat(
            [inputs.reset_index(drop=True), outputs.reset_index(drop=True)], axis=1
        )

        if trial.method.name.startswith("guidance"):
            QUESTION_CAT = [
                "Implementation Issues",
                "Feature Requests",
                "Concept Explanations",
                "Code Optimization",
                "Security and Privacy Concerns",
                "Model Training and Fine-tuning",
                "Data Handling and Manipulation",
                "User Interaction Flow",
                "Technical Integration",
                "Error Handling and Logging",
                "Customization and Configuration",
                "External API and Data Source Integration",
                "Language and Localization",
                "Streaming and Real-time Processing",
                "Tool Development",
                "Function Calling",
                "LLM Integrations",
                "General Agent Question",
                "General Chit Chat",
                "Memory",
                "Debugging Help",
                "Application Design",
                "Prompt Templates",
                "Cost Tracking",
                "Other",
            ]
            RESPONSE_TYPE = [
                "resolve issue",
                "provide guidance",
                "request information",
                "give up",
                "none",
                "other",
            ]

            @guidance(stateless=True, dedent=False)
            def guidance_list(lm):
                return (
                    lm
                    + "["
                    + zero_or_more(gen(regex=r'"[\w ]+", '))
                    + gen(regex=r'"[\w ]+"')
                    + "]"
                )

            WORD_PAT = r"[\w ]+"
            NEW_REC = "\n            "
            DOUBLE_QUOTE = '"'
            NEW_LINE = "\n"

            @guidance(stateless=False, dedent=False)
            def gen_chat_json(lm):
                lm += f"""{{
                    "GenerateTicket": {{
                        "issue_summary": "{gen(regex=WORD_PAT, stop='"')}",
                        "question": {{
                            "question_category": "{select(QUESTION_CAT, name='question_cat')}",
                            """
                if lm["question_cat"] == "Other":
                    lm += f""""category_if_other": "{gen(regex=WORD_PAT, stop='"')}",
                            """

                lm += f""""is_off_topic": {select(["false", "true"])},
                            "toxicity": {select([0, 1, 2, 3, 4, 5])},
                            "sentiment": "{select(["Negative", "Neutral", "Positive"])}",
                            "programming_language": "{select(["python", "javascript", "typescript", "unknown", "other"])}"
                        }},
                        "response": {{
                            "response_type": "{select(RESPONSE_TYPE, name='response_type')}",
                            """

                if lm["response_type"] == "other":
                    lm += f""""response_type_if_other": "{gen(regex=WORD_PAT, stop='"')}",
                            """

                lm += f""""confidence_level": {select([0, 1, 2, 3, 4, 5])}"""
                lm += f"""{select(['', ',' + NEW_REC + '"followup_actions":'], name='follow_up')}"""
                follow_up = lm.get("follow_up", None)
                if follow_up is not None and follow_up != '':
                    lm += f""" {guidance_list()}"""

                lm += f"""
                    }}
                }}
            }}"""
                return lm

        for i, row in merged_df.iterrows():
            # Initialize LLM
            if i == 0:
                if "mistral" in trial.method.name:
                    lm_path = hf_hub_download(
                        "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                        "mistral-7b-instruct-v0.2.Q8_0.gguf",
                    )
                elif "llama2-7b" in trial.method.name:
                    lm_path = hf_hub_download(
                        "TheBloke/Llama-2-7B-32K-Instruct-GGUF",
                        "llama-2-7b-32k-instruct.Q8_0.gguf",
                    )
                elif "phi-3" in trial.method.name:
                    lm_path = hf_hub_download(
                        "microsoft/Phi-3-mini-4k-instruct-gguf",
                        "Phi-3-mini-4k-instruct-fp16.gguf",
                    )
                else:
                    raise ValueError(
                        f"No support for method {trial.method.name}"
                    )  # pragma: no cover

                base_lm = models.LlamaCpp(
                    lm_path,
                    n_ctx=8192,
                    n_gpu_layers=-1,
                    echo=False,
                    verbose=False,
                )

            # Execute LLM
            print(f"{trial.method.name}[{i}]")
            start_time = time()
            lm = base_lm
            lm.engine.reset_metrics()
            if "mistral" in trial.method.name:
                lm += f"""<s>[INST] {row['system_prompt']}\n{row['user_prompt']} [/INST]"""
            elif "llama" in trial.method.name:
                lm += f"""<s>[INST] <<SYS>>\n{row['system_prompt']}\n<</SYS>>\n\n{row['user_prompt']}[/INST]"""
            elif "phi" in trial.method.name:
                lm += f"""<s><|user|>{row['system_prompt']}\n{row['user_prompt']}<|end|><|assistant|>"""
            else:
                raise ValueError(
                    f"Cannot support {trial.method.name} for system prompts"
                )  # pragma: no cover

            before_idx = len(str(lm))
            if "guidance" in trial.method.name:
                lm += gen_chat_json()
            else:
                lm += gen(max_tokens=1500)
            output_str = str(lm)[before_idx:]
            end_time = time()
            elapsed_time = end_time - start_time

            # Basic measures
            trial.log("output", output_str)
            trial.log("wall_time", elapsed_time)

            # Token statistics
            tm = {
                "input": lm.engine.metrics.engine_input_tokens,
                "output": lm.engine.metrics.engine_output_tokens,
                "token_count": lm.token_count,
            }
            tm["token_reduction"] = 1 - (tm["output"]) / (lm.token_count)
            trial.log("token_input", tm["input"])
            trial.log("token_output", tm["output"])
            trial.log("token_count", tm["token_count"])
            trial.log("token_reduction", tm["token_reduction"])

            # Validate JSON conformance
            json_success = False
            try:
                output_json = json.loads(output_str.strip())
                output_json = output_json["GenerateTicket"]
                json_success = True
            except Exception as e:
                trial.log("json_errmsg", str(e))
            trial.log("json", json_success * 1)

            if json_success:
                trial.log("output_json", output_json)
                trial.log("json_dirty", 0)
            else:
                success = False
                candidate = output_str.strip()
                for i, ch in enumerate(candidate):
                    if ch == "{":
                        try:
                            results = json_stream.load(io.StringIO(candidate[i:]))
                            di = json_stream.to_standard_types(results)
                            di = di["GenerateTicket"]
                            success = True
                            break
                        except Exception:
                            pass
                if success:
                    output_json = di
                    trial.log("output_json", output_json)
                    trial.log("json_dirty", 1)
                else:
                    trial.log("output_json", {})
                    trial.log("json_dirty", 0)

            # Validate JSON schema conformance
            from langchain_benchmarks.extraction.tasks.chat_extraction.schema import (
                GenerateTicket,
            )

            try:
                GenerateTicket.parse_obj(output_json)
                trial.log("json_valid", 1)
                if json_success:
                    trial.log("json_valid_strict", 1)
                else:
                    trial.log("json_valid_strict", 0)
            except Exception as e:
                trial.log("json_valid", 0)
                trial.log("json_valid_strict", 0)
                trial.log("json_valid_errmsg", str(e))

            # Toxicity similarity
            expected_json = row["output"]["output"]
            expected = expected_json["question"]["toxicity"]
            try:
                pred = output_json["question"]["toxicity"]
                score = 1 - abs(expected - float(pred)) / 5
                trial.log("toxicity", score)
                trial.log("toxicity_strict", score)
            except Exception as e:
                trial.log("toxicity_strict", 0)
                trial.log("toxicity_errmsg", str(e))

            # Sentiment similarity
            expected = expected_json["question"]["sentiment"]
            ordinal_map = {
                "negative": 0,
                "neutral": 1,
                "positive": 2,
            }
            expected_score = ordinal_map.get(str(expected).lower())
            try:
                pred = output_json["question"]["sentiment"]
                pred_score = ordinal_map.get(str(pred).lower())
                score = 1 - (abs(expected_score - float(pred_score)) / 2)
                trial.log("sentiment", score)
                trial.log("sentiment_strict", score)
            except Exception as e:
                trial.log("sentiment_strict", 0)
                trial.log("sentiment_errmsg", str(e))

            # Question category similarity
            expected = expected_json["question"]["question_category"]
            try:
                pred = output_json["question"]["question_category"]
                score = int(expected == pred)
                trial.log("question_cat", score)
                trial.log("question_cat_strict", score)
            except Exception as e:
                trial.log("question_cat_strict", 0)
                trial.log("question_cat_errmsg", str(e))

            # Off-topic similarity
            expected = expected_json["question"]["is_off_topic"]
            try:
                pred = output_json["question"].get("is_off_topic")
                score = int(expected == pred)
                trial.log("offtopic", score)
                trial.log("offtopic_strict", score)
            except Exception as e:
                trial.log("offtopic_strict", 0)
                trial.log("offtopic_errmsg", str(e))

            # Programming language similarity
            expected = expected_json["question"]["programming_language"]
            try:
                pred = output_json["question"]["programming_language"]
                score = int(expected == pred)
                trial.log("programming", score)
                trial.log("programming_strict", score)
            except Exception as e:
                trial.log("programming_strict", 0)
                trial.log("programming_errmsg", str(e))


def bench(
    db_url: str,
    experiment_name: str,
    models: List[str],
    force_recreate: bool,
    timeout: int,
    cache_dir: Union[str, Path],
    debug_mode: bool,
) -> Tuple[object, object]:
    """Runs the benchmark.

    Requires LANGCHAIN_API_KEY to be set as an environment variable the first time.

    Args:
        db_url (str): Database connection string.
        experiment_name (str): Name of experiment to create / run.
        models (List[str]): Models to benchmark.
        force_recreate (bool): Recreate the database before benchmarking.
        timeout (int): Max execution time per trial.
        cache_dir (Union[str, Path]): Cache to store external datasets.
        debug_mode (bool): Set this when you require a debugger to step line by line in the trial runner.

    Returns:
        Tuple[object, object]: (status, results) data frames where status relates to trials, results are wide form aggregates of each model.
    """

    from powerlift.bench import Benchmark, Store, populate_with_datasets
    from powerlift.executors import LocalMachine

    store = Store(db_url, force_recreate=force_recreate)
    populate_with_datasets(
        store, retrieve_langchain(cache_dir=cache_dir), exist_ok=True
    )
    executor = LocalMachine(store, n_cpus=1, debug_mode=debug_mode)

    bench = Benchmark(store, name=experiment_name)
    bench.run(
        langchain_chat_extract_runner,
        lambda x: langchain_chat_extract_filter_template(models, x),
        timeout=timeout,
        executor=executor,
    )

    bench.wait_until_complete()
    status_df = bench.status()
    result_df = bench.results()
    agg_df = result_df.pivot_table(
        index="method", columns="name", values="num_val", aggfunc=["mean", "std"]
    )
    agg_df.columns = ["_".join(x) for x in agg_df.columns.to_flat_index()]

    return status_df, agg_df
