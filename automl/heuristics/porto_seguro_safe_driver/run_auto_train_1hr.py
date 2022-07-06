import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import porto_seguro_safe_driver
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

porto_df = porto_seguro_safe_driver.load()
porto_seguro_safe_driver_df = get_repeatable_train_val_test_split(porto_df, 'target', random_seed=42)

auto_train_results = auto_train(
    dataset=porto_seguro_safe_driver_df,
    target='target',
    time_limit_s=3600,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
