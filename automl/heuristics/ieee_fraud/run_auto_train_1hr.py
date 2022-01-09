import logging
import pprint

from ludwig.datasets import ieee_fraud
from ludwig.automl import auto_train

ieee_fraud_df = ieee_fraud.load()

auto_train_results = auto_train(
    dataset=ieee_fraud_df,
    target='isFraud',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
