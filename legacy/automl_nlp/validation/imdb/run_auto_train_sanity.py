import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import imdb
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

i_df = imdb.load(split=False)
imdb_df = get_repeatable_train_val_test_split(i_df, 'sentiment', random_seed=42)

auto_train_results = auto_train(
    dataset=imdb_df,
    target='sentiment',
    time_limit_s=360,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
