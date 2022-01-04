import logging
import pprint

from ludwig.datasets import higgs
from ludwig.automl import create_auto_config

higgs_df = higgs.load()

auto_config = create_auto_config(
    dataset=higgs_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False,
    use_reference_config=True
)

pprint.pprint(auto_config)
