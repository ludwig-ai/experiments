import logging
import pprint

from ludwig.datasets import santander_customer_satisfaction
from ludwig.automl import create_auto_config

santander_customer_satisfaction_df = santander_customer_satisfaction.load()

auto_config = create_auto_config(
    dataset=santander_customer_satisfaction_df,
    target='TARGET',
    time_limit_s=86400,
    tune_for_memory=False
)

pprint.pprint(auto_config)
