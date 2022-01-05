import logging
import pprint

from load_util import load_ames_housing
from ludwig.automl import auto_train

ames_housing_df = load_ames_housing()

auto_train_results = auto_train(
    dataset=ames_housing_df,
    target='SalePrice',
    time_limit_s=3600,
    tune_for_memory=False,
    output_directory='s3://predibase-runs/nodeless/ames_housing/hours1/',
    user_config={'hyperopt': {'executor': {'max_concurrent_trials': 3, 'gpu_resources_per_trial': 1, 'cpu_resources_per_trial': 4}}},
)

pprint.pprint(auto_train_results)
