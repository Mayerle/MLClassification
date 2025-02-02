import pandas as pd
import numpy as np
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
    index_of = lambda class_: np.argwhere(class_ == unique_classes)[0][0]
    return np.array(list(map(index_of,classes)))
    
def label_encode(classes: np.ndarray, target_class) -> np.ndarray:
    index_of = lambda class_: 1 if class_ == target_class else -1
    return np.array(list(map(index_of,classes)))
        
def split_data(features: np.ndarray, targets:np.ndarray, test_size: float = 0.7, validate_size: float = 0.3) -> list[np.ndarray]:
    test_n   = math.trunc(test_size*features.shape[0])
    validate_n   = math.trunc(test_size*validate_size*features.shape[0])
    
    x_test     = features[:test_n]
    x_validate = x_test[validate_n:]
    x_test     = x_test[:validate_n]
    x_train    = features[test_n:]
    
    y_test     = targets[:test_n]
    y_validate = y_test[validate_n:]
    y_test     = y_test[:validate_n]
    y_train    = targets[test_n:]
    
    
    return [x_train, x_validate, x_test, y_train, y_validate, y_test]

def train_test_split(df: type[pd.DataFrame | np.ndarray], test_size: float = 0.3) -> list[pd.DataFrame | np.ndarray]:
    test_n   = math.trunc(test_size*df.shape[0])
    train = df[test_n:]
    test  = df[:test_n]
    return [train, test]

def train_validate_test_split(df: type[pd.DataFrame | np.ndarray], test_size: float = 0.3, validate_size: float = 0.3) -> type[pd.DataFrame | np.ndarray]:
    test_n   = math.trunc(test_size*df.shape[0])
    validate_n   = math.trunc(test_size*validate_size*df.shape[0])
    train = df[test_n:]
    test  = df[:test_n]
    validate = test[:validate_n]
    test  = test[validate_n:]
    return [train, validate, test]