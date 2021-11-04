import logging

from ludwig.api import LudwigModel
from ludwig.datasets import otto_group_product

model = LudwigModel(
    config='config_tabnet_reference_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

otto_group_product_df = otto_group_product.load()
model.train(
    dataset=otto_group_product_df,
    experiment_name='otto_group_product_tabnet_reference_laptop',
    model_name='otto_group_product_tabnet_reference_laptop',
    skip_save_processed_input=True
)
