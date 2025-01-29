import pandas as pd
from instruments.dftools import *

features_columns = ["SepalLengthCm","SepalWidthCm", "PetalLengthCm",  "PetalWidthCm"]
target_column = "Species"


df = pd.read_csv("dataset/iris.csv")
features = df[features_columns]
targets = df[target_column]
features = normalize_features(features)
features = convert_to_onevectors(features)


x_train, x_test, y_train, y_test = split_df(features,targets)

print(x_train, x_test, y_train, y_test)
