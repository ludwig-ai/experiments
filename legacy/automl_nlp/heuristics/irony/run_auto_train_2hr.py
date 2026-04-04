import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import irony
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

i_df = irony.load()
irony_df = get_repeatable_train_val_test_split(i_df, 'label', random_seed=42)

auto_train_results = auto_train(
    dataset=irony_df,
    target='label',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'output_features': [{'column': 'label', 'name': 'label', 'type': 'category'}],
        'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
