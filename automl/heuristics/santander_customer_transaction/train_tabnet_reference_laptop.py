import logging

from ludwig.api import LudwigModel
from ludwig.datasets import santander_customer_transaction

model = LudwigModel(
    config='config_tabnet_reference_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

santander_customer_transaction_df = santander_customer_transaction.load()
model.train(
    dataset=santander_customer_transaction_df,
    experiment_name='santander_customer_transaction_tabnet_reference_laptop',
    model_name='santander_customer_transaction_tabnet_reference_laptop',
    skip_save_processed_input=True
)
