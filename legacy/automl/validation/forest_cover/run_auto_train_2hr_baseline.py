import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import forest_cover
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

forest_df = forest_cover.load(use_tabnet_split=True)
forest_cover_df = get_repeatable_train_val_test_split(forest_df, 'Cover_Type', random_seed=42)

auto_train_results = auto_train(
    dataset=forest_cover_df,
    target='Cover_Type',
    time_limit_s=7200,
    tune_for_memory=False,
    output_directory='s3://predibase-elotl/baseline/forest_cover/hours2/',
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
