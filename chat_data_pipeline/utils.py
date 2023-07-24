import random
import re
from functools import partial
from collections import Counter

from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
import pandas as pd
import tqdm
import yaml

from chat_data_pipeline.pipeline import Pipeline, logger
from chat_data_pipeline import cleaners as cln
from chat_data_pipeline import filters as ftr
from chat_data_pipeline.kenlm_model import KenlmModel


def load_yaml(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_cleaners_from_config(config):
    cleaner_funcs = []
    cleaners = {}
    if config.get("cleaners") is not None:
        cleaners = config.get("cleaners", {})
    for cleaner, do_clean in cleaners.items():
        if do_clean:
            cleaner_funcs.append(
                getattr(cln, cleaner)
            )
    return cleaner_funcs


def get_filters_from_config(config):
    filter_funcs = []
    filters = {}
    if config.get("filters") is not None:
        filters = config.get("filters", {})
    for filter, value in filters.items():
        args = {}
        if value is not None:
            args = value.get("args", {})
        filter_func = custom_partial(
            getattr(ftr, filter),
            **args
        )
        filter_funcs.append(filter_func)
    return filter_funcs


def get_output_text_cleaners():
    cleaners = [
        cln.normalize_whitespace,
        cln.normalize_punctuation,
        cln.fix_utf8_encoding,
        cln.remove_empty_lines
    ]
    return cleaners


def get_input_text_cleaners():
    cleaners = [
        cln.normalize_whitespace,
        cln.remove_empty_lines
    ]
    return cleaners


def get_output_text_filters(filter_nsfw, filter_perplexity):
    filters = [
        custom_partial(
            ftr.check_word_number,
            min_word_threshold=5,
            max_word_threshold=128,
        ),
        custom_partial(
            ftr.check_completion,
        ),
        custom_partial(
            ftr.check_char_repetition,
            char_repetition_len=10,
            char_repetition_threshold=0.2,
        ),
        custom_partial(
            ftr.check_lowercase_ratio,
            lowercase_threshold=0.75,
        ),
    ]
    if filter_nsfw:
        filters.append(
            custom_partial(
                ftr.check_nsfw_words,
                flagged_words_threshold=0.025,
            ),
        )
    if filter_perplexity:
        filters.append(
            custom_partial(
                ftr.check_perplexity,
                kenlm_model=_get_kenlm_model(),
                min_perplexity_threshold=300,
                max_perplexity_threshold=10_000
            )
        )
    return filters


def _get_kenlm_model():
    kenlm_model = KenlmModel.from_pretrained(
        model_dataset="wikipedia",
        language="en",
        lower_case=True,
        remove_accents=True,
        normalize_numbers=True,
        punctuation=1,
    )
    return kenlm_model


def get_input_text_filters():
    filters = [
        custom_partial(
            ftr.check_lowercase_ratio,
            lowercase_threshold=0.55,
        ),
    ]
    return filters


def get_truncation_filters(splitter_token):
    filters = [
        custom_partial(
            ftr.check_truncation,
            splitter_token=splitter_token
        ),
    ]
    return filters


def custom_partial(func, **args):
    partial_func = partial(func, **args)
    partial_func.__name__ = func.__name__
    partial_func.__module__ = func.__module__
    return partial_func


def print_sample_dropped_examples(dataset, new_dataset, num_samples=5):
    original_ids = dataset["ids"]
    new_ids = new_dataset["ids"]
    dropped_ids = set(original_ids) - set(new_ids)
    num_samples = min(len(dropped_ids), num_samples)
    ids_to_show = random.sample(list(dropped_ids), num_samples)
    for id in ids_to_show:
        logger.info(f"Dropped sample: {dataset[id]}")


# Pipeline does not add column_name to newly added column with scores
def rename_dry_run_columns(dataset, filter_column_name):
    column_names = set(dataset.column_names)
    column_names = column_names - {"output_text", "input_text", "summary", "user_id"}
    columns_mapping = dict()
    for column_name in column_names:
        # Check if column already renamed by previous call of this function
        if "__" not in column_name:
            columns_mapping[column_name] = filter_column_name + "__" + column_name
    dataset = dataset.rename_columns(columns_mapping)
    return dataset


def get_edit_dataset(dataset_path):
    dataset = load_dataset(dataset_path, split="train", keep_in_memory=False)
    dataset = prepare_edit_dataset(dataset)
    return dataset


def prepare_edit_dataset(dataset):
    columns_mapping = {
        "model_input": "input_text",
        "edited_response": "output_text",
    }
    dataset = dataset.rename_columns(columns_mapping)
    columns_to_keep = list(columns_mapping.values()) + ["user_id", "response"]
    columns_to_remove = set(dataset.column_names) - set(columns_to_keep)
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def remove_unused_columns(dataset):
    columns_to_keep = ["user_id", "input_text", "output_text"]
    columns_to_remove = set(dataset.column_names) - set(columns_to_keep)
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def post_process_output_text(dataset):
    df = dataset.to_pandas()
    func = lambda x: " " + cln.clean_new_lines(x["output_text"]) + "\n"
    df["output_text"] = df.progress_apply(func, axis=1)
    dataset = Dataset.from_pandas(df)
    return dataset


def sample_datasets(datasets, proportions, target_size):
    target_size = min(
        [target_size] + [len(dataset) / proportion for proportion, dataset in zip(proportions, datasets)]
    )
    sampled_datasets = []
    for proportion, dataset in zip(proportions, datasets):
        sample_proportion = (target_size * proportion) / len(dataset)
        sampled_dataset = sample_dataset(dataset, sample_proportion)
        sampled_datasets.append(sampled_dataset)
    merged_dataset = concatenate_datasets(sampled_datasets)
    return merged_dataset


def sample_dataset(dataset, size):
    df = dataset.to_pandas()
    grouped = df.groupby('user_id')
    sample_groups = []
    for _, sub_group in tqdm.tqdm(grouped):
        sample_groups.append(_get_sample_group(sub_group, size=size))

    df_subset = pd.concat(sample_groups)
    df_subset = df_subset.drop(['__index_level_0__'], axis=1, errors='ignore')
    dataset_subset = Dataset.from_pandas(df_subset)
    return dataset_subset


def _get_sample_group(group, size):
    # helps with sampling superusers and do not touch small groups
    if len(group) >= 5:
        num_samples = int(len(group) * size)
        group = group.sample(num_samples)
    return group


def split_dataset_by_filter(dataset, column_name, filter_func):
    dataset_length = len(dataset)
    ids = range(dataset_length)
    dataset = dataset.add_column("ids", ids)
    filtered_dataset = run_filter(dataset, column_name, filter_func, dry_run=False)

    difference_dataset = _dataset_subtraction(dataset, filtered_dataset)

    filtered_dataset = filtered_dataset.remove_columns("ids")
    difference_dataset = difference_dataset.remove_columns("ids")

    return filtered_dataset, difference_dataset


def run_filter(dataset, column_name, filter_func, dry_run):
    datasources = [
        {
            "dataset": dataset,
            "name": "dataset",
            "columns": [column_name],
            "filters": [filter_func],
            "cleaners": [],
        },
    ]
    pipeline = Pipeline(datasources)
    pipeline.run(dry_run=dry_run)
    filtered_dataset = pipeline.datasources[0]["dataset"]
    return filtered_dataset


def run_cleaner(dataset, column_name, cleaners):
    datasources = [
        {
            "dataset": dataset,
            "name": "dataset",
            "columns": [column_name],
            "filters": [],
            "cleaners": cleaners,
        },
    ]
    pipeline = Pipeline(datasources)
    pipeline.run(dry_run=True)
    dataset = pipeline.datasources[0]["dataset"]
    return dataset


def _dataset_subtraction(minuend_dataset, subtrahend_dataset):
    original_ids = minuend_dataset["ids"]
    filtered_ids = subtrahend_dataset["ids"]
    dropped_ids = set(original_ids) - set(filtered_ids)
    original_df = minuend_dataset.to_pandas()
    difference_df = original_df[original_df.ids.isin(dropped_ids)]
    difference_df = difference_df.drop(['__index_level_0__'], axis=1, errors='ignore')
    difference_dataset = Dataset.from_pandas(difference_df)
    return difference_dataset


def add_concatenated_column(dataset, column_name, special_token):
    dataframe = dataset.to_pandas()
    func = lambda x: x["response"] + special_token + x["output_text"]
    dataframe[column_name] = dataframe.progress_apply(func, axis=1)
    dataset = Dataset.from_pandas(dataframe)
    return dataset


def get_words(text):
    return re.findall(r'\w+', text.lower())


# Adapted from:
# https://github.com/CarperAI/squeakily/blob/ba81f6e11fab424794d46cbf06d398ea2ad4a7f1/squeakily/filter.py#L81
def get_char_repetition_ratio(doc, char_rep_len):
    freq_char_ngrams = _get_frequency_ngrams(
        doc, char_rep_len
    )
    if len(freq_char_ngrams) == 0:
        return 0
    char_rep_ratio = _calculate_char_repetition_ratio(freq_char_ngrams)
    return char_rep_ratio


def _calculate_char_repetition_ratio(freq_char_ngrams):
    freq_char_ngrams = list(freq_char_ngrams.values())
    freq_char_ngrams = sorted(freq_char_ngrams, reverse=True)
    val_one = len([el for el in freq_char_ngrams if el == 1])
    num_rep_char_ngrams = min(
        int(np.sqrt(len(freq_char_ngrams))),
        len(freq_char_ngrams) - val_one,
    )
    char_rep_ratio = sum(
        freq_char_ngrams[:num_rep_char_ngrams]
    ) / sum(freq_char_ngrams)
    return char_rep_ratio


def _get_frequency_ngrams(doc, n):
    char_ngrams = [
        doc[i: i + n] for i in range(len(doc) - n + 1)
    ]
    freq_char_ngrams = Counter(char_ngrams)
    return freq_char_ngrams
