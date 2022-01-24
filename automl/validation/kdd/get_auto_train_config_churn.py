import logging
import pprint

from ludwig.datasets import kdd_churn
from ludwig.automl import create_auto_config

kdd_churn_df = kdd_churn.load()

auto_config = create_auto_config(
    dataset=kdd_churn_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
