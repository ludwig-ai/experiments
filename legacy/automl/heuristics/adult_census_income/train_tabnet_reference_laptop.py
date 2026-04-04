import logging

from ludwig.api import LudwigModel
from ludwig.datasets import adult_census_income
import pandas as pd

model = LudwigModel(
    config='config_tabnet_reference_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

adult_census_income_df = adult_census_income.load()
if "split" in adult_census_income_df.columns:
    train_df = adult_census_income_df[adult_census_income_df["split"] == 0]
    val_df = adult_census_income_df[adult_census_income_df["split"] == 1]
    test_df = adult_census_income_df[adult_census_income_df["split"] == 2]

    # no validation set provided, sample 10% of train set
    if len(val_df) == 0:
        val_df = train_df.sample(frac=0.1, replace=False)
        train_df = train_df.drop(val_df.index)

    val_df.split = 1
    adult_census_income_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

model.train(
    dataset=adult_census_income_df,
    experiment_name='adult_census_income_tabnet_reference_laptop',
    model_name='adult_census_income_tabnet_reference_laptop',
    skip_save_processed_input=True
)
