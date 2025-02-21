import pandas as pd
from instruments.dftools import *

def get_data():
    df = pd.read_csv("dataset/iris.csv").sample(frac=1,random_state=24)
    features_columns = ["SepalLengthCm","SepalWidthCm", "PetalLengthCm",  "PetalWidthCm"]
    target_column = "Species"

    objects = df[features_columns].to_numpy()
    targets = df[target_column].to_numpy()
    
    train_objects, test_objects = train_test_split(objects)
    train_targets, test_targets = train_test_split(targets)

    train_objects = normalize_features(train_objects)
    test_objects  = normalize_features(test_objects)

    train_targets = one_hot_encode(train_targets)
    test_targets = one_hot_encode(test_targets)
    return train_objects, test_objects, train_targets, test_targets 