import logging

from ludwig.api import LudwigModel
from ludwig.datasets import mushroom_edibility
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

model = LudwigModel(
    config='config_tabnet_transfer_otto.yaml',
    logging_level=logging.INFO,
    backend="local",
)

mushroom_df = mushroom_edibility.load()
mushroom_edibility_df = get_repeatable_train_val_test_split(mushroom_df, 'class', random_seed=42)
model.train(
    dataset=mushroom_edibility_df,
    experiment_name='mushroom_edibility_tabnet_transfer_otto',
    model_name='mushroom_edibility_tabnet_transfer_otto',
    skip_save_processed_input=True
)
