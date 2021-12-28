import pandas as pd
import numpy as np


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def transform(self, df, columns: list, inplace=False):
        self.mean = df.mean(numeric_only=True)
        self.std = df.std(numeric_only=True)
        if not inplace:
            df_copy = df.copy(deep=True)
        else:
            df_copy = df
        for i in columns:
            df_copy[i] = df_copy[i].apply(lambda x: (x - self.mean[i]) / self.std[i])
        return df_copy, self.mean, self.std

    def inverse_transform(self, df_list, columns: list, inplace=False):
        df_list = [df_list] if isinstance(df_list, pd.DataFrame) else df_list
        res = []
        for i in df_list:
            if not inplace:
                df_copy = i.copy(deep=True)
            else:
                df_copy = i
            for i in columns:
                df_copy[i] = df_copy[i].apply(lambda x: (x * self.std[i]) + self.mean[i])
                res.append(df_copy)
        return res