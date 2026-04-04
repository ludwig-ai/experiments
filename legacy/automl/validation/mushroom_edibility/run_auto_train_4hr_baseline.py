import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import mushroom_edibility
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

mushroom_df = mushroom_edibility.load()
mushroom_edibility_df = get_repeatable_train_val_test_split(mushroom_df, 'class', random_seed=42)

auto_train_results = auto_train(
    dataset=mushroom_edibility_df,
    target='class',
    time_limit_s=14400,
    tune_for_memory=False,
    output_directory='s3://predibase-elotl/baseline/mushroom_edibility/hours4/',
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
