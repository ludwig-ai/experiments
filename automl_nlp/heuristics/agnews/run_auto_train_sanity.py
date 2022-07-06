import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import agnews
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

a_df = agnews.load(split=False)
a_df.drop("title", axis=1, inplace=True)
if "class" in a_df.columns:
    a_df.drop("class", axis=1, inplace=True)
agnews_df = get_repeatable_train_val_test_split(a_df, 'class_index', random_seed=42)

auto_train_results = auto_train(
    dataset=agnews_df,
    target='class_index',
    time_limit_s=360,
    tune_for_memory=True,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
