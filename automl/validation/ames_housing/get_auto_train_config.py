import logging
import pprint

from ludwig.datasets import ames_housing
from ludwig.automl import create_auto_config

ames_housing_df = ames_housing.load()

auto_config = create_auto_config(
    dataset=ames_housing_df,
    target='SalePrice',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
