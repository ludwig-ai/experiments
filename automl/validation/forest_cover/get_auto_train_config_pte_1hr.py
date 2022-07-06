import logging
import pprint

from ludwig.automl import create_auto_config
from ludwig.datasets import forest_cover
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

forest_df = forest_cover.load(use_tabnet_split=True)
forest_cover_df = get_repeatable_train_val_test_split(forest_df, 'Cover_Type', random_seed=42)

auto_config = create_auto_config(
    dataset=forest_cover_df,
    target='Cover_Type',
    time_limit_s=3600,
    tune_for_memory=False,
    user_config={'preprocessing': {'split': {'column': 'split', 'type': 'fixed'}},
        'hyperopt': {'sampler': {'search_alg': {'type': 'random', 'points_to_evaluate': [{
        'combiner.bn_momentum': 0.6,
        'combiner.bn_virtual_bs': 4096,
        'combiner.num_steps': 4,
        'combiner.output_size': 128,
        'combiner.relaxation_factor': 1.2,
        'combiner.size': 32,
        'combiner.sparsity': 0.000001,
        'training.batch_size': 4096,
        'training.decay_rate': 0.9,
        'training.decay_steps': 20000,
        'training.learning_rate': 0.01
        }]}}}}
)

pprint.pprint(auto_config)
