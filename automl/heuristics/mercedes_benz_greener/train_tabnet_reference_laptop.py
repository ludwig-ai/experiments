import logging

from ludwig.api import LudwigModel
from ludwig.datasets import mercedes_benz_greener

model = LudwigModel(
    config='config_tabnet_reference_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

mercedes_benz_greener_df = mercedes_benz_greener.load()
model.train(
    dataset=mercedes_benz_greener_df,
    experiment_name='mercedes_benz_greener_tabnet_reference_laptop',
    model_name='mercedes_benz_greener_tabnet_reference_laptop',
    skip_save_processed_input=True
)
