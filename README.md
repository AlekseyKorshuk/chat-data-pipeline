# Chat Data Pipeline

This repository helps to clean, filter and deduplicate conversation datasets.

## Quick Start

Clone and install dependencies:

```shell
git clone https://github.com/AlekseyKorshuk/chat-data-pipeline
cd chat-data-pipeline
pip install -r requirements.txt
```

We will prepare very small dataset of instructions:

```shell
python3 main.py --config_path ./experiments/tiny-example.yaml
```

You can take a look at the YAML file to discover the structure of the config.

Initial dataset has the following structure of one sample:

```json
{
  "conversation": [
    {
      "content": "Explain the main differences between an alligator and a crocodile.",
      "do_train": false,
      "role": "User"
    },
    {
      "content": "Alligators and crocodiles belong to the same order, Crocodilia, but they have several differences. 1) Shape of the snout: Alligators have a U-shaped, wider snout, while crocodiles have a more pointed, V-shaped snout. 2) Teeth placement: In an alligator, lower teeth are mostly hidden when its mouth is closed, while in a crocodile, the fourth lower tooth is visible even when the mouth is closed. 3) Habitat: Alligators are mostly found in freshwater habitats such as swamps and rivers, while crocodiles can be found in both freshwater and saltwater habitats. 4) Distribution: Alligators are mainly found in the southeastern United States and parts of China, whereas crocodiles have a more widespread distribution across Africa, Asia, the Americas, and Australia.",
      "do_train": true,
      "role": "Assistant"
    }
  ]
}
```

This example could have more conversation turns: User, Assistant, User, Assistant...

As well role can be "System" at the very first item in the list.

# Custom Setup

In general, you can use this for any dataset that has a string column. Here is an example usage:

```python
from datasets import load_dataset

from chat_data_pipeline import utils
from chat_data_pipeline.preprocessor import DataPreprocessor
from chat_data_pipeline import cleaners as cln
from chat_data_pipeline import filters as ftr

dataset = load_dataset("AlekseyKorshuk/tiny-imdb", split="train")

deduplication_config = {
    'do_deduplication': True,
    'minhash_config': {
        'ngram_size': 5,
        'num_perm': 256,
        'threshold': 0.7,
        'min_ngram_size': 5
    }
}

cleaners = [cln.fix_utf8_encoding, cln.normalize_punctuation, cln.remove_empty_lines]
filters = [
    utils.custom_partial(ftr.check_word_number,
                         min_word_threshold=0,
                         max_word_threshold=10000),
]

preprocessor = DataPreprocessor(
    dataset=dataset,
    column_name="text",
    cleaners=cleaners,
    filters=filters,
    deduplication_config=deduplication_config,
    verbose=False,
)
preprocessed_dataset = preprocessor.run()
```

## Acknowledgment

This is a friendly fork of Squeakily by CarperAI, but this repository aims only on conversation data, uses pandas to
speed up the pipeline and latest near deduplication. 
