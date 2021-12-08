import logging
import pprint

from load_util import load_synthetic_fraud
from ludwig.automl import auto_train

synthetic_fraud_df = load_synthetic_fraud()

auto_train_results = auto_train(
    dataset=synthetic_fraud_df,
    target='isFraud',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
