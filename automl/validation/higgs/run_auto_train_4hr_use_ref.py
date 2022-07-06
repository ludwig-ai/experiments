import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import higgs
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

h_df = higgs.load()
higgs_df = get_repeatable_train_val_test_split(h_df, 'label', random_seed=42)

auto_train_results = auto_train(
    dataset=higgs_df,
    target='label',
    time_limit_s=14400,
    tune_for_memory=False,
    use_reference_config=True,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
