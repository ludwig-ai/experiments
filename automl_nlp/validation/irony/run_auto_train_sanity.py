import logging
import pprint

from load_util import load_irony
from ludwig.automl import auto_train

irony_df = load_irony()

auto_train_results = auto_train(
    dataset=irony_df,
    target='label',
    time_limit_s=360,
    tune_for_memory=False,
    user_config={'output_features': [{'column': 'label', 'name': 'label', 'type': 'category'}]}
)

pprint.pprint(auto_train_results)
