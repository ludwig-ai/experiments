import logging

from ludwig.api import LudwigModel
from load_util import load_mushroom_edibility

model = LudwigModel(
    config='config_tabnet_transfer_otto.yaml',
    logging_level=logging.INFO,
    backend="local",
)

mushroom_edibility_df = load_mushroom_edibility()
model.train(
    dataset=mushroom_edibility_df,
    experiment_name='mushroom_edibility_tabnet_transfer_otto',
    model_name='mushroom_edibility_tabnet_transfer_otto',
    skip_save_processed_input=True
)
