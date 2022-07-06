import logging

from ludwig.api import LudwigModel
from ludwig.datasets import ieee_fraud
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

model = LudwigModel(
    config='config_tabnet_imbalance.yaml',
    logging_level=logging.INFO,
    backend="local",
)

ieee_df = ieee_fraud.load()
ieee_fraud_df = get_repeatable_train_val_test_split(ieee_df, 'isFraud', random_seed=42)
model.experiment(
    dataset=ieee_fraud_df,
    experiment_name='ieee_fraud_tabnet_imbalance',
    model_name='ieee_fraud_tabnet_imbalance',
    skip_save_processed_input=True
)
