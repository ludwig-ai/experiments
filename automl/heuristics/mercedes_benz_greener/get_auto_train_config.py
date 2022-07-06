import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import mercedes_benz_greener
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

mercedes_df = mercedes_benz_greener.load()
mercedes_benz_greener_df = get_repeatable_train_val_test_split(mercedes_df, random_seed=42)

auto_config = create_auto_config(
    dataset=mercedes_benz_greener_df,
    target='y',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_config)
