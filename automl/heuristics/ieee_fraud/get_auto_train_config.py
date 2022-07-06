import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import ieee_fraud
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

ieee_df = ieee_fraud.load()
ieee_fraud_df = get_repeatable_train_val_test_split(ieee_df, 'isFraud', random_seed=42)

auto_config = create_auto_config(
    dataset=ieee_fraud_df,
    target='isFraud',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
