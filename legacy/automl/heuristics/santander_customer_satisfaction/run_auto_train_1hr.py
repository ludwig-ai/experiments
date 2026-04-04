import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import santander_customer_satisfaction
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

santander_df = santander_customer_satisfaction.load()
santander_customer_satisfaction_df = get_repeatable_train_val_test_split(santander_df, 'TARGET', random_seed=42)

auto_train_results = auto_train(
    dataset=santander_customer_satisfaction_df,
    target='TARGET',
    time_limit_s=3600,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
