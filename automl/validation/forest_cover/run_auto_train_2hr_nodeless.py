import logging
import pprint

from ludwig.datasets import forest_cover
from ludwig.automl import auto_train

forest_cover_df = forest_cover.load()

auto_train_results = auto_train(
    dataset=forest_cover_df,
    target='Cover_Type',
    time_limit_s=7200,
    tune_for_memory=False,
    output_directory='s3://predibase-runs/nodeless/forest_cover/hours2/',
    user_config={'hyperopt': {'executor': {'max_concurrent_trials': 3}, 'sampler': {'search_alg': {'type': 'hyperopt', 'random_state_seed': 42}}}},
)

pprint.pprint(auto_train_results)
