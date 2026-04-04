import logging

from ludwig.api import LudwigModel
from ludwig.datasets import higgs
from ludwig.utils.dataset_utils import get_repeatable_train_val_test_split

model = LudwigModel(
    config='config_tabnet_transfer_santCustSat.yaml',
    logging_level=logging.INFO,
    backend="local",
)

h_df = higgs.load()
higgs_df = get_repeatable_train_val_test_split(h_df, 'label', random_seed=42)
model.train(
    dataset=higgs_df,
    experiment_name='higgs_tabnet_transfer_santCustSat',
    model_name='higgs_tabnet_transfer_santCustSat',
    skip_save_processed_input=True
)
