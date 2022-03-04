import pandas as pd

from ludwig.datasets import agnews

def load_agnews(include_title: bool = False):
    agnews_df = agnews.load(split=False)

    # Concatenate title and description to produce single column for Text AutoML
    if include_title:
        agnews_df["description"] = agnews_df["title"] + " " + agnews_df["description"]

    agnews_df.drop("title", axis=1, inplace=True)

    if "split" in agnews_df.columns:
        train_df = agnews_df[agnews_df["split"] == 0]
        val_df = agnews_df[agnews_df["split"] == 1]
        test_df = agnews_df[agnews_df["split"] == 2]

        # no validation set provided, sample 10% of train set
        if len(val_df) == 0:
            val_df = train_df.sample(frac=0.1, replace=False, random_state=42)
            train_df = train_df.drop(val_df.index)

        val_df.split = 1
        agnews_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return agnews_df
