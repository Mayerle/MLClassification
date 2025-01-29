import pandas as pd
import numpy as np
import scipy.stats
import math


def normalize_features(features: pd.DataFrame)-> pd.DataFrame:
    get_std = lambda x: np.std(x)
    means = features.apply(np.mean, axis=0)
    stds = features.apply(get_std, axis=0)
    features = features.sub(means,axis=1).div(stds,axis=1)
    return features

def convert_to_onevectors(features: pd.DataFrame):
    new_df = pd.DataFrame()
    FEATURES_COLUMN_NAME = "Features"
    concat_one = lambda x: np.concatenate((x,[1]))
    new_df[FEATURES_COLUMN_NAME] = features.apply(concat_one, axis=1)
    return new_df

def split_df(features: pd.DataFrame, targets:pd.DataFrame, train_size: float = 0.7, seed: float = 1) -> list[np.ndarray]:
    features  = features.sample(frac=1,random_state=seed)
    targets   = features.sample(frac=1,random_state=seed)
    
    train_n   = math.trunc(train_size*features.shape[0])
    x_train   = features.iloc[:train_n,:].to_numpy()
    x_test    = features.iloc[train_n:,:].to_numpy()
    
    y_train   = targets.iloc[:train_n,:].to_numpy()
    y_test    = targets.iloc[train_n:,:].to_numpy()
    
    return [x_train,  x_test, y_train,  y_test]
