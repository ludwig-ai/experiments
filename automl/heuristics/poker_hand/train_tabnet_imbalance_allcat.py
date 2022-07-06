import logging

from ludwig.api import LudwigModel
from ludwig.datasets import poker_hand
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

model = LudwigModel(
    config='config_tabnet_imbalance_allcat.yaml',
    logging_level=logging.INFO,
    backend="local",
)

poker_df = poker_hand.load(split=False)
poker_hand_df = get_repeatable_train_val_test_split(poker_df, 'hand', random_seed=42)
model.experiment(
    dataset=poker_hand_df,
    experiment_name='poker_hand_tabnet_imbalance_allcat',
    model_name='poker_hand_tabnet_imbalance_allcat',
    skip_save_processed_input=True
)
