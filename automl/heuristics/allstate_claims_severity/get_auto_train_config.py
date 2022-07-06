import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import allstate_claims_severity
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

allstate_df = allstate_claims_severity.load()
allstate_claims_severity_df = get_repeatable_train_val_test_split(allstate_df, random_seed=42)

auto_config = create_auto_config(
    dataset=allstate_claims_severity_df,
    target='loss',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
