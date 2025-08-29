import pandas as pd
import csv

fname = "xgboost.csv"

df = pd.read_csv(f"{fname}", skiprows=1)

        # then, if you want to overwrite the original:
df.to_csv(fname, index=False)

