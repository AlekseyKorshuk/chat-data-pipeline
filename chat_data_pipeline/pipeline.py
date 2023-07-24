import logging

import numpy as np
from datasets import Dataset, concatenate_datasets
from rich.logging import RichHandler
import tqdm

tqdm.tqdm.pandas()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(RichHandler(rich_tracebacks=True))
# Turn off logging for datasets
logging.getLogger("datasets").setLevel(logging.ERROR)


class Pipeline:
    def __init__(self, datasources):
        self.datasources = datasources

    def run(self, dry_run=False):
        for i in range(len(self.datasources)):
            self.datasources[i]["dataset"] = self.datasources[i]["dataset"].to_pandas()

            column_name = self.datasources[i]["columns"][0]
            logger.info(f"Running datasource: {self.datasources[i]['name']}")

            for cleaner_func in self.datasources[i]["cleaners"]:
                self.datasources[i]["dataset"] = apply_cleaner(
                    self.datasources[i]["dataset"],
                    column_name,
                    cleaner_func
                )

            for filter_func in self.datasources[i]["filters"]:
                self.datasources[i]["dataset"] = apply_filter(
                    self.datasources[i]["dataset"],
                    column_name,
                    filter_func,
                    dry_run
                )
            self.datasources[i]["dataset"] = smart_from_pandas(self.datasources[i]["dataset"])


def apply_cleaner(dataframe, column_name, cleaner_func):
    logger.info(f"Running cleaner: {cleaner_func.__name__} on {column_name}")
    func = lambda x: cleaner_func(x[column_name])
    dataframe[column_name] = dataframe.progress_apply(func, axis=1)
    return dataframe


def apply_filter(dataframe, column_name, filter_func, dry_run):
    logger.info(f"Running filter: {filter_func.__name__} on {column_name}")
    criteria_column_name = f"{column_name}_{filter_func.__name__}_criteria"
    func = lambda x: filter_func(x[column_name], dry_run=dry_run)
    dataframe[criteria_column_name] = dataframe.progress_apply(func, axis=1)
    logger.info(f"Criteria statistics:\n{dataframe[criteria_column_name].describe()}")
    if not dry_run:
        func = lambda x: x[criteria_column_name]
        dataframe = dataframe[dataframe.progress_apply(func, axis=1)]
        dataframe = dataframe.drop(
            [criteria_column_name, "__index_level_0__"],
            axis=1,
            errors='ignore'
        )

    return dataframe


def smart_from_pandas(df, chunk_size=200_000):
    datasets = []
    for g, batch in df.groupby(np.arange(len(df)) // chunk_size):
        dataset = Dataset.from_pandas(batch, preserve_index=False)
        datasets.append(dataset)
    return concatenate_datasets(datasets)
