import logging

from ludwig.api import LudwigModel
from ludwig.datasets import poker_hand

model = LudwigModel(
    config='config_tabnet_reference_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

poker_hand_df, _, _ = poker_hand.load()
model.train(
    dataset=poker_hand_df,
    experiment_name='poker_hand_tabnet_reference_laptop',
    model_name='poker_hand_tabnet_reference_laptop',
    skip_save_processed_input=True
)
