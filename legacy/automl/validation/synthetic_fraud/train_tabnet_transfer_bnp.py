import logging

from ludwig.api import LudwigModel
from ludwig.datasets import synthetic_fraud
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

model = LudwigModel(
    config='config_tabnet_transfer_bnp.yaml',
    logging_level=logging.INFO,
    backend="local",
)

synthetic_df = synthetic_fraud.load()
synthetic_fraud_df = get_repeatable_train_val_test_split(synthetic_df, 'isFraud', random_seed=42)
model.train(
    dataset=synthetic_fraud_df,
    experiment_name='synthetic_fraud_tabnet_transfer_bnp',
    model_name='synthetic_fraud_tabnet_transfer_bnp',
    skip_save_processed_input=True
)
