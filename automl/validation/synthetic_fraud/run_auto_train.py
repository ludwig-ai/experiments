import logging
import pprint

from ludwig.datasets import synthetic_fraud
from ludwig.automl import auto_train

synthetic_fraud_df = synthetic_fraud.load()

# NOTE: export FUNCTION_SIZE_ERROR_THRESHOLD="1000000000"
auto_train_results = auto_train(
    dataset=synthetic_fraud_df,
    target='isFraud',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
