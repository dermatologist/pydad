import pandas as pd 
import os.path

class Dad(object):

    def __init__ (filepath: str, version: str):
        self._filepath = filepath
        self._version = version

    def read_sample(self):
        if os.path.exists(self._filepath + clin_sample_spss.fth):
            self._dfs = pd.read_feather(self._filepath + clin_sample_spss.fth)
        else:
            self._dfs = pd.read_spss(self._filepath + clin_sample_spss.sav)
            self._dfs.to_feather(self._filepath + clin_sample_spss.fth)

    def read_full(self):
        if os.path.exists(self._filepath + self._version +'.fth'):
            self._df = pd.read_feather(self._filepath + self._version +'.fth')
        else:
            self._df = pd.read_spss(self._filepath + self._version +'.sav')
            self._df.to_feather(self._filepath + self._version +'.fth')
