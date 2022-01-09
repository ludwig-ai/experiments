import logging
import pprint

from load_util import load_adult_census_income
from ludwig.automl import auto_train

adult_census_income_df = load_adult_census_income()

auto_train_results = auto_train(
    dataset=adult_census_income_df,
    target='income',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
