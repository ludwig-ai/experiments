import logging
import pprint

from ludwig.datasets import synthetic_fraud
from ludwig.automl import create_auto_config

synthetic_fraud_df = synthetic_fraud.load()

auto_config = create_auto_config(
    dataset=synthetic_fraud_df,
    target='isFraud',
    time_limit_s=86400,
    tune_for_memory=False
)

pprint.pprint(auto_config)
