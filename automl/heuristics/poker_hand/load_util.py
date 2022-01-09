import pandas as pd

from ludwig.datasets import poker_hand

def load_poker_hand():
    train_df, test_df, val_df = poker_hand.load()

    # no validation set provided, sample 10% of train set
    if len(val_df) == 0:
        val_df = train_df.sample(frac=0.1, replace=False, random_state=42)
        train_df = train_df.drop(val_df.index)

    train_df['split'] = 0
    val_df['split'] = 1
    test_df['split'] = 2
    poker_hand_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    return poker_hand_df
