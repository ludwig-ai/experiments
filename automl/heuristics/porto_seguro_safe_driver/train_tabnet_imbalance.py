import logging

from ludwig.api import LudwigModel
from load_util import load_porto_seguro_safe_driver

model = LudwigModel(
    config='config_tabnet_imbalance.yaml',
    logging_level=logging.INFO,
    backend="local",
)

porto_seguro_safe_driver_df = load_porto_seguro_safe_driver()
model.experiment(
    dataset=porto_seguro_safe_driver_df,
    experiment_name='porto_seguro_safe_driver_tabnet_imbalance',
    model_name='porto_seguro_safe_driver_tabnet_imbalance',
    skip_save_processed_input=True
)
