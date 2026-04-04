import logging

from ludwig.api import LudwigModel
from ludwig.datasets import porto_seguro_safe_driver

model = LudwigModel(
    config='config_transf_sanity_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

porto_seguro_safe_driver_df = porto_seguro_safe_driver.load()
model.train(
    dataset=porto_seguro_safe_driver_df,
    experiment_name='porto_seguro_safe_driver_transf_sanity_laptop',
    model_name='porto_seguro_safe_driver_transf_sanity_laptop',
    skip_save_processed_input=True
)
