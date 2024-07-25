import pandas as pd
import numpy as np


house_df = pd.read_csv("data/train.csv")
house_df

price_mean = house_df['SalePrice'].mean()

sample_df = pd.read_csv("data/sample_submission.csv")
sample_df

sample_df['SalePrice'] = price_mean

sample_df.to_csv('/data/sample_submission.csv', index=False)
