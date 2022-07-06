import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import forest_cover
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

forest_df = forest_cover.load(use_tabnet_split=True)
forest_cover_df = get_repeatable_train_val_test_split(forest_df, 'Cover_Type', random_seed=42)

auto_train_results = auto_train(
    dataset=forest_cover_df,
    target='Cover_Type',
    time_limit_s=3600,
    tune_for_memory=False,
    output_directory='s3://predibase-runs/nodeless/forest_cover/hours1/',
    user_config={'hyperopt': {'executor': {'max_concurrent_trials': 3, 'gpu_resources_per_trial': 1, 'cpu_resources_per_trial': 4}},
        'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
