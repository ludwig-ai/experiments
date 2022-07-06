import logging

from ludwig.api import LudwigModel
from ludwig.datasets import walmart_recruiting
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

model = LudwigModel(
    config='config_tabnet_imbalance.yaml',
    logging_level=logging.INFO,
    backend="local",
)

walmart_df = walmart_recruiting.load()
walmart_recruiting_df = get_repeatable_train_val_test_split(walmart_df, 'TripType', random_seed=42)
model.experiment(
    dataset=walmart_recruiting_df,
    experiment_name='walmart_recruiting_tabnet_imbalance',
    model_name='walmart_recruiting_tabnet_imbalance',
    skip_save_processed_input=True
)
