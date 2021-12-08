import logging
import pprint

from load_util import load_ames_housing
from ludwig.automl import auto_train

ames_housing_df = load_ames_housing()

auto_train_results = auto_train(
    dataset=ames_housing_df,
    target='SalePrice',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
