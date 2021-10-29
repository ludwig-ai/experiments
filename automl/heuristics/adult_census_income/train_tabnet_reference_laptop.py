import logging

from ludwig.api import LudwigModel
from ludwig.datasets import adult_census_income

model = LudwigModel(
    config='config_tabnet_reference_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

adult_census_income_df = adult_census_income.load()
model.train(
    dataset=adult_census_income_df,
    experiment_name='adult_census_income_tabnet_reference_laptop',
    model_name='adult_census_income_tabnet_reference_laptop',
    skip_save_processed_input=True
)
