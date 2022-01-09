import logging
import pprint

from ludwig.datasets import bnp_claims_management
from ludwig.automl import auto_train

bnp_claims_management_df = bnp_claims_management.load()

auto_train_results = auto_train(
    dataset=bnp_claims_management_df,
    target='target',
    time_limit_s=3600,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)
