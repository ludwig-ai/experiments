import logging
import pprint

from ludwig.datasets import adult_census_income
from ludwig.automl import create_auto_config

adult_census_income_df = adult_census_income.load()

auto_config = create_auto_config(
    dataset=adult_census_income_df,
    target='income',
    time_limit_s=86400,
    tune_for_memory=False
)

pprint.pprint(auto_config)
