"""Backend integration for benchmarking.

Currently supports powerlift.
"""

from typing import Generator, Optional
import pathlib


def retrieve_langchain(cache_dir: Optional[str] = None) -> Generator[object, None, None]:
    """Retrieves LangChain datasets appropriate for guidance benchmarking. Requires env `LANGCHAIN_API_KEY` to be set.

    Args:
        cache_dir (Optional[str], optional): Directory to store downloaded datasets. Defaults to None.

    Yields:
        Generator[object, None, None]: DataFrameDataset required for powerlift.
    """

    import pandas as pd
    from langchain_benchmarks import registry
    from langsmith.client import Client
    from langchain_benchmarks import clone_public_dataset
    from powerlift.bench.store import update_cache, retrieve_cache, DataFrameDataset
    import os

    if os.getenv("LANGCHAIN_API_KEY") is None:
        raise ValueError("Env variable LANGCHAIN_API_KEY is not set.")

    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir, "langchain")

    name = "chat_extract"
    inputs_name = f"{name}.inputs.parquet"
    outputs_name = f"{name}.outputs.parquet"
    meta_name = f"{name}.meta.json"
    cached = retrieve_cache(cache_dir, [inputs_name, outputs_name, meta_name])
    if cached is None:
        task = registry["Chat Extraction"]
        clone_public_dataset(task.dataset_id, dataset_name=task.name)
        client = Client()
        dataset = client.read_dataset(dataset_name=task.name)
        examples = list(client.list_examples(dataset_id=dataset.id, as_of=None))

        system_prompt = "You are a data extraction bot tasked with " + \
                        "extracting and inferring information from dialogues and generating tickets. Always respond " + \
                        f"only with json based on the following JSON schema:\n{task.schema.schema_json()}"

        user_prompts = []
        for example in examples:
            dialogue = f"<question>\n{example.inputs['question']}\n</question>\n" + \
                        f"<assistant-response>\n{example.inputs['answer']}\n</assistant-response>"
            user_prompt = "Generate a ticket from the following question-response pair:\n" + \
                            f"<Dialogue>\n{dialogue}\n</Dialogue>\n" + \
                            "Remember, respond directly with this format:\n" + \
                            f'{{"{task.schema.schema()["title"]}": ...}}\n' + \
                            "RESPOND ONLY IN JSON THEN STOP."
            user_prompts.append(user_prompt)

        inputs = pd.DataFrame.from_records([{
            "input": example.inputs,
            "schema": task.schema.schema_json(),
            "system_prompt": system_prompt,
            "user_prompt": user_prompts[i],
        } for i, example in enumerate(examples)])

        outputs = pd.DataFrame.from_records([{"output": example.outputs} for example in examples])
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