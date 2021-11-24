import logging
import pprint

from ludwig.datasets import allstate_claims_severity
from ludwig.automl import create_auto_config

allstate_claims_severity_df = allstate_claims_severity.load()

auto_config = create_auto_config(
    dataset=allstate_claims_severity_df,
    target='loss',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
