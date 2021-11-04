import logging

from ludwig.api import LudwigModel
from ludwig.datasets import porto_seguro_safe_driver

model = LudwigModel(
    config='config_tabnet_reference_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

porto_seguro_safe_driver_df = porto_seguro_safe_driver.load()
model.train(
    dataset=porto_seguro_safe_driver_df,
    experiment_name='porto_seguro_safe_driver_tabnet_reference_laptop',
    model_name='porto_seguro_safe_driver_tabnet_reference_laptop',
    skip_save_processed_input=True
)
