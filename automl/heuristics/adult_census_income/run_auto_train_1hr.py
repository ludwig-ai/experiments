import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import adult_census_income
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

adult_df = adult_census_income.load()
adult_census_income_df = get_repeatable_train_val_test_split(adult_df, 'income', random_seed=42)

auto_train_results = auto_train(
    dataset=adult_census_income_df,
    target='income',
    time_limit_s=3600,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
