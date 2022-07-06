import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import otto_group_product
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

otto_df = otto_group_product.load()
otto_group_product_df = get_repeatable_train_val_test_split(otto_df, 'target', random_seed=42)

auto_train_results = auto_train(
    dataset=otto_group_product_df,
    target='target',
    time_limit_s=7200,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
