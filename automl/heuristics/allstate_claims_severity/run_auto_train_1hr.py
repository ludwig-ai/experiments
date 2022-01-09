import logging
import pprint

from ludwig.datasets import allstate_claims_severity
from ludwig.automl import auto_train

allstate_claims_severity_df = allstate_claims_severity.load()

auto_train_results = auto_train(
    dataset=allstate_claims_severity_df,
    target='loss',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
