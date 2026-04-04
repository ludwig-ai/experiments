import logging

from ludwig.api import LudwigModel
from ludwig.datasets import walmart_recruiting

model = LudwigModel(
    config='config_concat_sanity_laptop.yaml',
    logging_level=logging.INFO,
    backend="local",
)

walmart_recruiting_df, _, _ = walmart_recruiting.load()
model.train(
    dataset=walmart_recruiting_df,
    experiment_name='walmart_recruiting_concat_sanity_laptop',
    model_name='walmart_recruiting_concat_sanity_laptop',
    skip_save_processed_input=True
)
