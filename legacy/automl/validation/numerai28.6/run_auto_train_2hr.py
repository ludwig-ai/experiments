import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import numerai28pt6
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

numer_df = numerai28pt6.load()
numerai28pt6_df = get_repeatable_train_val_test_split(numer_df, 'target', random_seed=42)

auto_train_results = auto_train(
    dataset=numerai28pt6_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
