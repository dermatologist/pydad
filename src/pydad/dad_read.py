import pandas as pd 
import numpy as np
import functools

class DadRead(object):

    def __init__ (self, df):
        self._df = df

    def has_diagnosis(self, diagnosis):
        mask = functools.reduce(np.logical_or, [self._df['D_I10_{}'.format(i)].str.startswith(diagnosis) for i in range(1, 25)])
        return self._df.loc[mask]

    def has_treatment(self, treatment):
        mask = functools.reduce(np.logical_or, [self._df['I_CCI_{}'.format(i)].str.startswith(treatment) for i in range(1, 20)])
        return self._df.loc[mask]

    @staticmethod
    def count(df):
        index = df.index
        return len(index)