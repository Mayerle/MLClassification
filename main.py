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
target_labels = targets.unique()
objects = normalize_features(objects)
objects = convert_to_onevectors(objects)
targets = one_hot_encode(targets)

x_train, x_test, y_train, y_test = split_data(objects,targets)


#Quadric model
quadric_model = QuadricCModel()
quadric_model.fit(x_train,y_train)
quadric_predictions = quadric_model.predict(x_test)
quadric_stats = ClassificationStatistics(y_test,quadric_predictions)
matrix = quadric_stats.calculate_confusion_matrix()


#Logit model
logit_model = LogisticCModel()
logit_model.fit(x_train,y_train)
logit_predictions = logit_model.predict(x_test)
logit_stats = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats.calculate_confusion_matrix()


#SVM model
svm_model = SVMCModel()
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()

statistics = {
    'Quadric': quadric_stats.calculate_all(),
    'Logit': logit_stats.calculate_all(),
    'SVM': svm_stats.calculate_all(),
}

plot_all(statistics, 2)
plt.show()