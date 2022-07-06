import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import synthetic_fraud
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

synthetic_df = synthetic_fraud.load()
synthetic_fraud_df = get_repeatable_train_val_test_split(synthetic_df, 'isFraud', random_seed=42)

auto_config = create_auto_config(
    dataset=synthetic_fraud_df,
    target='isFraud',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
