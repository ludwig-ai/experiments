import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import bnp_claims_management
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

bnp_df = bnp_claims_management.load()
bnp_claims_management_df = get_repeatable_train_val_test_split(bnp_df, 'target', random_seed=42)

auto_config = create_auto_config(
    dataset=bnp_claims_management_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
