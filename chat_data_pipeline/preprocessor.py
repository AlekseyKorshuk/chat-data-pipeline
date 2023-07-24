import gc
import shutil

from datasets import Dataset, load_from_disk

from chat_data_pipeline.pipeline import logger
from chat_data_pipeline import utils
from chat_data_pipeline.minhash_deduplication import deduplicate


class DataPreprocessor:
    dataset: Dataset

    def __init__(
            self,
            dataset,
            column_name,
            cleaners,
            filters,
            deduplication_config,
            dry_run=False,
            verbose=False
    ):
        self.dataset = dataset
        self.column_name = column_name
        self.cleaners = cleaners
        self.filters = filters
        self.deduplication_config = deduplication_config
        self.dry_run = dry_run
        self.verbose = verbose

    def run(self):
        self._clean_dataset()
        self._filter_dataset()
        if self.deduplication_config.get("do_deduplication", False):
            self._deduplicate_dataset()
        return self.dataset

    def _clean_dataset(self):
        if len(self.cleaners) > 0:
            self.dataset = utils.run_cleaner(self.dataset, self.column_name, self.cleaners)
        return self.dataset

    def _filter_dataset(self):
        for filter_func in self.filters:
            dataset_length = len(self.dataset)
            ids = range(dataset_length)
            self.dataset = self.dataset.add_column("ids", ids)
            filtered_dataset = utils.run_filter(
                dataset=self.dataset,
                column_name=self.column_name,
                filter_func=filter_func,
                dry_run=self.dry_run
            )
            self._print_filter_logs(filtered_dataset, filter_func.__name__)
            self.dataset = filtered_dataset.remove_columns("ids")

        return self.dataset

    def _deduplicate_dataset(self):
        dataset_length = len(self.dataset)
        ids = range(dataset_length)
        self.dataset = self.dataset.add_column("ids", ids)
        # need to save to disk and load again, otherwise it is very slow
        target_directory = "./.temp-dataset"
        shutil.rmtree(target_directory, ignore_errors=True)
        try:
            self.dataset.save_to_disk(target_directory)
        except PermissionError:
            logger.info("Can not save dataset, nothing changed. Skipping...")
        gc.collect()
        self.dataset = load_from_disk(target_directory)
        deduplicated_ds = deduplicate(
            self.dataset,
            column=self.column_name,
            **self.deduplication_config.get("args", {})
        )
        self.dataset = deduplicated_ds.remove_columns("ids")
        return self.dataset

    def _print_filter_logs(self, filtered_dataset, filter_name):
        original_length = len(self.dataset)
        filtered_length = len(filtered_dataset)
        reduced_percent = round(100 * (original_length - filtered_length) / original_length, 2)
        logger.info(
            f'Filtered by {filter_name} on {self.column_name}:\n'
            f'{reduced_percent}% = {original_length - filtered_length:,} samples reduced\n'
            f'New dataset size: {filtered_length:,} rows'
        )
        if self.verbose:
            utils.print_sample_dropped_examples(self.dataset, filtered_dataset, num_samples=10)
