import pandas as pd

from ludwig.datasets import amazon_reviews

def load_amazon_reviews(include_title: bool = False):
    amazon_reviews_df = amazon_reviews.load(split=False)

    # Concatenate title and text to produce single column for Text AutoML
    if include_title:
        amazon_reviews_df["review_text"] = amazon_reviews_df["review_tile"] + " " + amazon_reviews_df["review_text"]

    amazon_reviews_df.drop("review_tile", axis=1, inplace=True)

    if "split" in amazon_reviews_df.columns:
        train_df = amazon_reviews_df[amazon_reviews_df["split"] == 0]
        val_df = amazon_reviews_df[amazon_reviews_df["split"] == 1]
        test_df = amazon_reviews_df[amazon_reviews_df["split"] == 2]

        # no validation set provided, sample 10% of train set
        if len(val_df) == 0:
            val_df = train_df.sample(frac=0.1, replace=False, random_state=42)
            train_df = train_df.drop(val_df.index)

        val_df.split = 1
        amazon_reviews_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return amazon_reviews_df
