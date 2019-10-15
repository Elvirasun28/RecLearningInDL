import tensorflow as tf
import pandas as pd
import numpy as np
from DCN.DCN import DCN
from DCN.util import FeatureDictionary,DataParser
import DCN.config as config
from sklearn.model_selection import StratifiedKFold

def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ['id','target']]
        df['missing_feat'] = np.sum((df[cols] == -1).values,axis = 1)
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"] * df["ps_reg_03"]
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ['id','target']]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain['target'].values
    X_test = dfTest[cols].values
    y_test = dfTest['target'].values

    return dfTrain,dfTest,X_train,y_train,X_test,y_test


def run_base_model_dcn(dfTrain,dfTest,folds,dcn_params):





