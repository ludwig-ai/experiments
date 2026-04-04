import logging
import pprint

from ludwig.automl import auto_train
from ludwig.datasets import mnist
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split
 
m_df = mnist.load(split=False)
mnist_df = get_repeatable_train_val_test_split(m_df, 'label', random_seed=42)

auto_train_results = auto_train(
    dataset=mnist_df,
    target='label',
    time_limit_s=360,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}}},
)

pprint.pprint(auto_train_results)
