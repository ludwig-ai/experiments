import pandas as pd

from ludwig.datasets import goemotions

def load_goemotions():
    goemotions_df = goemotions.load(split=False)
    goemotions_df.drop("comment_id", axis=1, inplace=True)
    return goemotions_df
