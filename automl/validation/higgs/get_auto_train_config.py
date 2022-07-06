import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import higgs
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

h_df = higgs.load()
higgs_df = get_repeatable_train_val_test_split(h_df, 'label', random_seed=42)

auto_config = create_auto_config(
    dataset=higgs_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
