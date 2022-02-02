import logging
import pprint

from load_util import load_connect4
from ludwig.automl import auto_train

connect4_df = load_connect4()

auto_train_results = auto_train(
    dataset=connect4_df,
    target='winner',
    time_limit_s=3600,
    tune_for_memory=False,
    use_reference_config=True,
    user_config={'output_features': [{'column': 'winner', 'name': 'winner', 'type': 'category'}]}
)

pprint.pprint(auto_train_results)
