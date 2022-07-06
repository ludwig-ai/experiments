import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import adult_census_income
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

adult_df = adult_census_income.load()
adult_census_income_df = get_repeatable_train_val_test_split(adult_df, 'income', random_seed=42)

auto_config = create_auto_config(
    dataset=adult_census_income_df,
    target='income',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
