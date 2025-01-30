import pandas as pd
from instruments.models import *
from instruments.dftools import *
from instruments.statistic import *
from instruments.view import *

#features names and target name
features_columns = ["SepalLengthCm","SepalWidthCm", "PetalLengthCm",  "PetalWidthCm"]
target_column = "Species"

#sample data
df = pd.read_csv("dataset/iris.csv").sample(frac=1,random_state=1)
objects = df[features_columns]
targets = df[target_column]
objects = normalize_features(objects)
objects = convert_to_onevectors(objects)
targets = one_hot_encode(targets)

x_train, x_test, y_train, y_test = split_data(objects,targets)

#Quadric model
quadric_model = ClassificationModel("quadric")
quadric_model.fit(x_train,y_train)
quadric_predictions = quadric_model.predict(x_test)
quadric_stats = ClassificationStatistics(y_test,quadric_predictions)

matrix = quadric_stats.calculate_confusion_matrix()


#Logit model
logit_model = ClassificationModel("logit")
logit_model.fit(x_train,y_train)
logit_predictions = logit_model.predict(x_test)
logit_stats = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats.calculate_confusion_matrix()



#SVM model
svm_model = ClassificationModel("svm")
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()

#Print results
print("[Quadric model]")
print(quadric_stats.calculate_all())
print("[Logit model]")
print(logit_stats.calculate_all())
print("[SVM model]")
print(svm_stats.calculate_all())