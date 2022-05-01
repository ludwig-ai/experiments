import logging
import pprint

from load_util import load_ohsumed_7400
from ludwig.automl import auto_train

ohsumed_7400_df = load_ohsumed_7400()

auto_train_results = auto_train(
    dataset=ohsumed_7400_df,
    target='intent',
    time_limit_s=3600,
    tune_for_memory=True,
    output_directory='s3://predibase-runs/nodeless/ohsumed_7400/',
    user_config={'hyperopt': {'executor': {'max_concurrent_trials': 3, 'gpu_resources_per_trial': 1, 'cpu_resources_per_trial': 4}}},
)

pprint.pprint(auto_train_results)
