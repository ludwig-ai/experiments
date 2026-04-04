import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import imdb
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

i_df = imdb.load(split=False)
imdb_df = get_repeatable_train_val_test_split(i_df, 'sentiment', random_seed=42)

auto_config = create_auto_config(
    dataset=imdb_df,
    target='sentiment',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
