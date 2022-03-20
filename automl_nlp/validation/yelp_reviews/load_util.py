import pandas as pd

from ludwig.datasets import yelp_reviews

def load_yelp_reviews():
    yelp_reviews_df = yelp_reviews.load(split=False)

    if "split" in yelp_reviews_df.columns:
        train_df = yelp_reviews_df[yelp_reviews_df["split"] == 0]
        val_df = yelp_reviews_df[yelp_reviews_df["split"] == 1]
        test_df = yelp_reviews_df[yelp_reviews_df["split"] == 2]

        # no validation set provided, sample 10% of train set
        if len(val_df) == 0:
            val_df = train_df.sample(frac=0.1, replace=False, random_state=42)
            train_df = train_df.drop(val_df.index)

        val_df.split = 1
        yelp_reviews_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return yelp_reviews_df
