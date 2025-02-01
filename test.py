from instruments.dftools import *
import pandas as pd


df = pd.read_csv("dataset/iris.csv").sample(frac=1,random_state=1)

print(df.shape[0])

test, train = train_test_split(df)
print(test.shape[0],train.shape[0])


test, validate, train = train_validate_test_split(df)
print(test.shape[0],validate.shape[0],train.shape[0])