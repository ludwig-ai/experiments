import logging
import pprint

from ludwig.datasets import irony
from ludwig.automl import create_auto_config

irony_df = irony.load()

auto_config = create_auto_config(
    dataset=irony_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'output_features': [{'column': 'label', 'name': 'label', 'type': 'category'}]}
)

pprint.pprint(auto_config)
