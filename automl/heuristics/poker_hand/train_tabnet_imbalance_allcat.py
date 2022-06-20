import logging

from ludwig.api import LudwigModel
from load_util import load_poker_hand

model = LudwigModel(
    config='config_tabnet_imbalance_allcat.yaml',
    logging_level=logging.INFO,
    backend="local",
)

poker_hand_df = load_poker_hand()
model.experiment(
    dataset=poker_hand_df,
    experiment_name='poker_hand_tabnet_imbalance_allcat',
    model_name='poker_hand_tabnet_imbalance_allcat',
    skip_save_processed_input=True
)
