import logging

from ludwig.api import LudwigModel
from load_util import load_poker_hand

model = LudwigModel(
    config='config_tabnet_imbalance.yaml',
    logging_level=logging.INFO,
    backend="local",
)

poker_hand_df = load_poker_hand()
model.experiment(
    dataset=poker_hand_df,
    experiment_name='poker_hand_tabnet_imbalance',
    model_name='poker_hand_tabnet_imbalance',
    skip_save_processed_input=True
)
