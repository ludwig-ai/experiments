import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import reuters_r8
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

reuters_df = reuters_r8.load(split=False)
reuters_df.drop("edge", axis=1, inplace=True)
reuters_r8_df = get_repeatable_train_val_test_split(reuters_df, 'intent', random_seed=42)

auto_train_results = auto_train(
    dataset=reuters_r8_df,
    target='intent',
    time_limit_s=3600,
    tune_for_memory=True,
    output_directory='s3://predibase-elotl/baseline/reuters_r8/',
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
