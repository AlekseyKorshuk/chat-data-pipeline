import os

import click
from datasets import load_dataset, concatenate_datasets

from chat_data_pipeline.pipeline import logger
from chat_data_pipeline import utils
from chat_data_pipeline.preprocessor import DataPreprocessor

PAD = 32


@click.command()
@click.option('--config_path')
def main(config_path):
    config = utils.load_yaml(config_path)
    dataset_paths = [dataset["dataset_path"] for dataset in config["datasets"]]
    output_dataset_path = config["output_dataset_path"]
    verbose = config.get("verbose", False)

    instruction_config = config["instruction_config"]
    response_config = config["response_config"]

    dataset = combine_datasets(dataset_paths)

    dataset = dataset.map(
        convert_to_input_output,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=list(dataset.features),
        desc="Converring to I/O..."
    )

    dataset = dataset.map(
        add_content_columns,
        batched=False,
        num_proc=os.cpu_count(),
        desc="Adding content column..."
    )

    print(utils.get_cleaners_from_config(response_config))
    print(utils.get_filters_from_config(response_config))
    print(response_config.get("deduplication", {}))
    preprocessor = DataPreprocessor(
        dataset=dataset,
        column_name="response",
        cleaners=utils.get_cleaners_from_config(response_config),
        filters=utils.get_filters_from_config(response_config),
        deduplication_config=response_config.get("deduplication", {}),
        verbose=verbose,
    )
    dataset = preprocessor.run()

    cleaners = utils.get_cleaners_from_config(instruction_config)
    if len(cleaners) > 0:
        logger.warning("Cleaner does not work on instructions. Cleaners set to empty list.")
    preprocessor = DataPreprocessor(
        dataset=dataset,
        column_name="instruction",
        cleaners=[],
        filters=utils.get_filters_from_config(instruction_config),
        deduplication_config=instruction_config.get("deduplication", {}),
        verbose=verbose,
    )
    dataset = preprocessor.run()

    prepared_dataset_chatml = dataset.map(
        convert_to_chatml,
        batched=False,
        num_proc=os.cpu_count(),
        remove_columns=list(dataset.features)
    )
    prepared_dataset_chatml = prepared_dataset_chatml.shuffle(seed=42)
    prepared_dataset_chatml.push_to_hub(output_dataset_path)
    logger.info(prepared_dataset_chatml)


def combine_datasets(dataset_paths):
    datasets = []
    for dataset_path in dataset_paths:
        dataset = load_dataset(dataset_path)
        dataset = concatenate_datasets(list(dataset.values()))
        if "source" not in dataset.features:
            dataset = dataset.add_column("source", [dataset_path] * len(dataset))
        datasets.append(dataset)
    dataset = concatenate_datasets(datasets)
    return dataset


def convert_to_input_output(examples):
    sources = []
    inputs = []
    outputs = []
    for conversation, source in zip(examples["conversation"], examples["source"]):
        input = []
        for message in conversation:
            if message["do_train"]:
                inputs.append(input.copy())
                outputs.append(message)
                sources.append(source)
            input.append(message)
    return {
        "input": inputs,
        "output": outputs,
        "source": sources
    }


def add_content_columns(example):
    response = example["output"]["content"].strip()
    instruction = ""
    if len(example["input"]) > 0:
        instruction = example["input"][-1]["content"].strip()
    return {
        "instruction": instruction,
        "response": response,
    }


def convert_to_chatml(example):
    conversation = []
    for message in example["input"]:
        message["do_train"] = False
        conversation.append(message)
    conversation.append(
        {
            "content": example["response"],
            "role": example["output"]["role"],
            "do_train": True,
        }
    )
    return {
        "conversation": conversation,
        "source": example["source"]
    }


if __name__ == "__main__":
    main()
