import logging
import pprint

from ludwig.datasets import bnp_claims_management
from ludwig.automl import create_auto_config

bnp_claims_management_df = bnp_claims_management.load()

create_auto_config_results = create_auto_config(
    dataset=bnp_claims_management_df,
    target='target',
    time_limit_s=72,
    tune_for_memory=False
)

pprint.pprint(create_auto_config_results)
