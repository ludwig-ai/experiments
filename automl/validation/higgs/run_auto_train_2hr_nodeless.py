import logging
import pprint

from load_util import load_higgs
from ludwig.automl import auto_train

higgs_df = load_higgs()

auto_train_results = auto_train(
    dataset=higgs_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False,
    output_directory='s3://predibase-runs/nodeless/higgs/hours2/',
    user_config={'hyperopt': {'executor': {'max_concurrent_trials': 3, 'gpu_resources_per_trial': 1, 'cpu_resources_per_trial': 4}, 'sampler': {'search_alg': {'type': 'hyperopt', 'random_state_seed': 42}}}},
)

pprint.pprint(auto_train_results)
