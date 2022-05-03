import logging
import pprint

from load_util import load_bbcnews
from ludwig.automl import auto_train

bbcnews_df = load_bbcnews()

auto_train_results = auto_train(
    dataset=bbcnews_df,
    target='Category',
    time_limit_s=3600,
    tune_for_memory=True,
    output_directory='s3://predibase-runs/nodeless/bbcnews/',
    user_config={'hyperopt': {'executor': {'max_concurrent_trials': 3, 'gpu_resources_per_trial': 1, 'cpu_resources_per_trial': 4}}},
)

pprint.pprint(auto_train_results)
