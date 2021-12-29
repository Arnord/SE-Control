import pandas as pd
import numpy as np


class DataFrameScalar:
    def __init__(self):
        self.mean = pd.DataFrame()
        self.std = pd.DataFrame()

    def get_stats(self):
        return self.mean.values, self.std.values

    def set_stats(self, df):
        self.mean = df.mean(numeric_only=True)
        self.std = df.std(numeric_only=True)
        return self.mean.values, self.std.values

    def transform(self, df, stats_columns: list, inplace=False):
        self.set_stats(df)
        if inplace:
            df_copy = df
        else:
            df_copy = df.copy(deep=True)
        for j in stats_columns:
            df_copy[j] = df_copy[j].apply(lambda x: (x - self.mean[j]) / self.std[j])
        if not inplace:
            return df_copy

    def inverse_transform(self, df, columns: list, inplace=False):
        if inplace:
            df_copy = df
        else:
            df_copy = df.copy(deep=True)
        for j in columns:
            df_copy[j] = df_copy[j].apply(lambda x: (x * self.std[j]) + self.mean[j])
        if not inplace:
            return df_copy
