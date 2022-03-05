from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import logging
import pickle
import pandas as pd
import numpy as np
import warnings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")

file_handler = logging.FileHandler("Logs/model.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

warnings.filterwarnings('ignore')


class Model:
    def __init__(self, input_data):
        print("Created Model object.")
        logger.debug("Created Model object.")

        self.input_data = input_data
        self.columns = ['age', 'workclass', 'fnlwgt', 'education', 'marital-status','occupation', 'relationship', 'race', 'sex', 'capital-gain','capital-loss', 'hours-per-week', 'country']
        self.cat = ['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'sex', 'country']
        self.num = ['capital-gain', 'capital-loss', 'age', 'fnlwgt', 'hours-per-week']
        self.df = None
        self.scalar = pickle.load(open("pick/scalar.sav", "rb"))
        self.oe = pickle.load(open("pick/oe.sav", "rb"))
        self.dt = pickle.load(open("pick/tuned_dt_2.sav", "rb"))
        self.knn = pickle.load(open("pick/tuned_knn_2.sav", "rb"))
        self.rf = pickle.load(open("pick/tuned_rf_2.sav", "rb"))
        self.xgb = XGBClassifier()
        self.xgb.load_model("pick/tuned_xgb_4.bin")

    def generate_dataframe(self):
        try:
            print("Inside generate_dataframe. Generating DataFrame out of the input data.")
            logger.debug("Inside generate_dataframe. Generating DataFrame out of the input data.")

            input_df = pd.DataFrame(columns=self.columns)
            input_df.loc[0] = self.input_data
            self.df = input_df

            logger.debug("Exiting generate_dataframe successfully !!!")

        except Exception as e:
            print("Error occurred inside generate_dataframe of Model class.", e)
            logger.debug("Error occurred inside generate_dataframe of Model class.")
            raise e

    def preprocessor(self):
        try:
            print("Inside preprocessor. Preprocessing the input dataframe.")
            logger.debug("Inside preprocessor. Preprocessing the input dataframe.")

            self.df.loc[:, self.num] = self.scalar.transform(self.df[self.num])
            self.df.loc[:, self.cat] = self.oe.transform(self.df[self.cat])
            for i in self.num:
                self.df[i] = np.array(self.df[i], dtype='float64')

            logger.debug("Exiting preprocessor successfully !!!")

        except Exception as e:
            print("Error occurred inside preprocessor of Model class.", e)
            logger.debug("Error occurred inside preprocessor of Model class.")
            raise e

    def soft_voting_prediction(self):
        try:
            print("Inside soft_voting_prediction. Starting prediction.")
            logger.debug("Inside soft_voting_prediction. Starting prediction.")

            zeroes = []
            ones = []

            temp_predict = self.xgb.predict_proba(self.df.to_numpy())
            zeroes.append(temp_predict[0][0])
            ones.append(temp_predict[0][1])

            temp_predict = self.rf.predict_proba(self.df.to_numpy())
            zeroes.append(temp_predict[0][0])
            ones.append(temp_predict[0][1])

            temp_predict = self.knn.predict_proba(self.df.to_numpy())
            zeroes.append(temp_predict[0][0])
            ones.append(temp_predict[0][1])

            temp_predict = self.dt.predict_proba(self.df.to_numpy())
            zeroes.append(temp_predict[0][0])
            ones.append(temp_predict[0][1])

            zeroes = np.array(zeroes)
            ones = np.array(ones)

            y_proba = [zeroes.mean(), ones.mean()]
            print(y_proba)
            y_hat = y_proba.index(max(y_proba))

            logger.debug("Exiting soft_voting_prediction successfully !!!")

            return y_hat

        except Exception as e:
            print("Error occurred inside soft_voting_prediction of Model class.", e)
            logger.debug("Error occurred inside soft_voting_prediction of Model class.")
            raise e
