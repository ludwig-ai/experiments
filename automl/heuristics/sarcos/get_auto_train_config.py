import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import sarcos
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

s_df = sarcos.load(split=False)
sarcos_df = get_repeatable_train_val_test_split(s_df, 'torque_1', random_seed=42)

auto_config = create_auto_config(
    dataset=sarcos_df,
    target='torque_1',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
