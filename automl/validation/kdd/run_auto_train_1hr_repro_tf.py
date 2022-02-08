import logging
import pprint

from load_util import load_kdd_appetency
from ludwig.automl import auto_train

kdd_appetency_df = load_kdd_appetency()

auto_train_results = auto_train(
    dataset=kdd_appetency_df,
    target='target',
    time_limit_s=360,
    tune_for_memory=False,
    user_config={'hyperopt': {
        'executor': {'max_concurrent_trials': 1, 'gpu_resources_per_trial': 1, 'cpu_resources_per_trial': 1},
        'sampler': {'search_alg': {'points_to_evaluate': [{
            'combiner.bn_momentum': 0.7,
            'combiner.bn_virtual_bs': 2048,
            'combiner.num_steps': 4,
            'combiner.output_size': 8,
            'combiner.relaxation_factor': 1.0,
            'combiner.size': 32,
            'combiner.sparsity': 0.0001,
            'training.batch_size': 8192,
            'training.decay_rate': 0.95,
            'training.decay_steps': 500,
            'training.learning_rate': 0.025
            }]}}
        }}
)

pprint.pprint(auto_train_results)
