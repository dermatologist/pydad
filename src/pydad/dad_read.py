import pandas as pd 
import numpy as np
import functools
from sklearn.preprocessing import MultiLabelBinarizer

class DadRead(object):

    def __init__ (self, df):
        self._df = df
        self._df['morbidities'] = self._df[['D_I10_{}'.format(i) for i in range(1, 25)]].values.tolist()
        self._df['interventions'] = self._df[['I_CCI_{}'.format(i) for i in range(1, 20)]].values.tolist()
        self._demographics = self._df[['SUB_PROV', 'AGRP_F_D', 'GENDER', 'X_FR_I_T', 'ADM_CAT', 'ENT_CODE', 'X_TO_I_T', 'DIS_DISP', 'WGHT_GRP']]
        self._los = self._df[['TLOS_CAT', 'ACT_LCAT', 'ALC_LCAT']]
        self._los['TLOS_CAT_BIN'] = np.where(self._los['TLOS_CAT'] >=10, 1, 0)
        self._mlb = MultiLabelBinarizer()

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

    def vector(self, significant_chars=3, include_treatments=True):
        morbidities = self._df[['D_I10_{}'.format(i) for i in range(1, 25)]].str[:significant_chars].values.tolist()
        interventions = self._df[['I_CCI_{}'.format(i) for i in range(1, 20)]].str[:significant_chars].values.tolist()

        disease_vector = self._mlb.fit_transform(morbidities .dropna())
        # Column names of the dataframe are the class names of multilabel binarizer
        disease_df = pd.DataFrame(data = disease_vector, columns=mlb.classes_)
        treatment_vector = mlb.fit_transform(interventions.dropna())
        treatment_df = pd.DataFrame(data = treatment_vector, columns=mlb.classes_)
        if(include_treatments):
            horizontal_stack = pd.concat([self._demographics, disease_df, treatment_df, self._los], axis=1)
        else:
            horizontal_stack = pd.concat([self._demographics, disease_df, self._los], axis=1)
         # Remove the empty string and ZZZ
        if(significant_chars>0):
            z = 'Z' * flag
            empty = ' ' * flag
        else:
            z = 'ZZZZZZ'
            empty = "      "

        horizontal_stack = horizontal_stack.drop(columns=z)
        horizontal_stack = horizontal_stack.drop(columns=empty)
        return horizontal_stack