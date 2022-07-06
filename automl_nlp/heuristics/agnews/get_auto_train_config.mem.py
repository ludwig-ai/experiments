import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import agnews
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

a_df = agnews.load(split=False)
a_df.drop("title", axis=1, inplace=True)
if "class" in a_df.columns:
    a_df.drop("class", axis=1, inplace=True)
agnews_df = get_repeatable_train_val_test_split(a_df, 'class_index', random_seed=42)

auto_config = create_auto_config(
    dataset=agnews_df,
    target='class_index',
    time_limit_s=7200,
    tune_for_memory=True,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
