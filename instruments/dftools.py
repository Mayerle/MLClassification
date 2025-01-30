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

def convert_to_onevectors(features: pd.DataFrame)-> np.ndarray:
    concat_one = lambda x: np.concatenate((x,[1]))
    return features.apply(concat_one, axis=1).to_numpy()


def one_hot_encode(classes: np.ndarray) -> np.ndarray:
    unique_classes = np.unique(classes)
    index_of = lambda class_: np.argwhere(class_ == unique_classes)[0]
    return np.array(list(map(index_of,classes)))
    
def label_encode(classes: np.ndarray, target_class) -> np.ndarray:
    index_of = lambda class_: 1 if class_ == target_class else -1
    return np.array(list(map(index_of,classes)))
        
def split_data(features: np.ndarray, targets:np.ndarray, train_size: float = 0.7) -> list[np.ndarray]:
    train_n   = math.trunc(train_size*features.shape[0])
    x_train   = features[:train_n]
    x_test    = features[train_n:]
    
    y_train   = targets[:train_n]
    y_test    = targets[train_n:]
    
    return [x_train,  x_test, y_train,  y_test]
