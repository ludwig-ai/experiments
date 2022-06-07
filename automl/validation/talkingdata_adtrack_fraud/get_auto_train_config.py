import logging
import pprint

from load_util import load_talkingdata_adtrack_fraud
from ludwig.automl import create_auto_config

talkingdata_adtrack_fraud_df = load_talkingdata_adtrack_fraud()

auto_config = create_auto_config(
    dataset=talkingdata_adtrack_fraud_df,
    target='is_attributed',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_config)
