import logging
import pprint

from ludwig.datasets import bnp_claims_management
from ludwig.automl import create_auto_config

bnp_claims_management_df = bnp_claims_management.load()

auto_config = create_auto_config(
    dataset=bnp_claims_management_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
