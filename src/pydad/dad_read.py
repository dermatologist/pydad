import pandas as pd 
import numpy as np
import functools

class DadRead(object):

    def __init__ (self, df):
        self._df = df
        self._df['morbidities'] = self._df[['D_I10_{}'.format(i) for i in range(1, 25)]].values.tolist()
        self._df['interventions'] = self._df[['I_CCI_{}'.format(i) for i in range(1, 20)]].values.tolist()

    def has_diagnosis(self, diagnosis):
        mask = functools.reduce(np.logical_or, [self._df['D_I10_{}'.format(i)].str.startswith(diagnosis) for i in range(1, 25)])
        return self._df.loc[mask]

    def comorbidity(self, diagnosis):
        with_diagnosis = self.has_diagnosis(diagnosis)
        l = with_diagnosis['morbidities'].tolist()
        return self.list_to_dict(l)


    def interventions(self, treatment):
        with_treatment = self.has_treatment(treatment)
        l = with_treatment['interventions'].tolist()
        return self.list_to_dict(l)

    def list_to_dict(self, list_of_lists):
        """Flattens the list of lists and converts to key count dict

        Arguments:
            list_of_lists {[list]} -- [2D list]

        Returns:
            [dict] -- [key - count]
        """
        flat_list = [item for sublist in list_of_lists for item in sublist]
        counts = dict()
        for i in flat_list:
            counts[i] = counts.get(i, 0) + 1
        if '' in counts:
            del counts['']
        return counts        

    def has_treatment(self, treatment):
        mask = functools.reduce(np.logical_or, [self._df['I_CCI_{}'.format(i)].str.startswith(treatment) for i in range(1, 20)])
        return self._df.loc[mask]

    @staticmethod
    def count(df):
        index = df.index
        return len(index)