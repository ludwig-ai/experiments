import logging

from ludwig.api import LudwigModel
from ludwig.datasets import poker_hand

model = LudwigModel(
    config='config_tabnet_imbalance.yaml',
    logging_level=logging.INFO,
    backend="local",
)

poker_hand_df, _, _ = poker_hand.load()
model.experiment(
    dataset=poker_hand_df,
    experiment_name='poker_hand_tabnet_imbalance',
    model_name='poker_hand_tabnet_imbalance',
    skip_save_processed_input=True
)
