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
x_train, x_validate, x_test, y_train, y_validate, y_test = split_data(objects,targets)

#Fit models
knn_model0 = KNNClassificationModel(norm="l1",neighbors_n=3)
knn_model0.fit(x_train, y_train)
predictions = knn_model0.predict(x_test)
knn_stats0 = ClassificationStatistics(y_test, predictions)
matrix = knn_stats0.calculate_confusion_matrix()


knn_model1 = KNNClassificationModel(norm="l2",neighbors_n=3)
knn_model1.fit(x_train, y_train)
predictions = knn_model1.predict(x_test)
knn_stats1 = ClassificationStatistics(y_test, predictions)
matrix = knn_stats1.calculate_confusion_matrix()


knn_model2 = KNNClassificationModel(norm="l1",neighbors_n=20)
knn_model2.fit(x_train, y_train)
predictions = knn_model2.predict(x_test)
knn_stats2 = ClassificationStatistics(y_test, predictions)
matrix = knn_stats2.calculate_confusion_matrix()


knn_model3 = KNNClassificationModel(norm="l2",neighbors_n=20)
knn_model3.fit(x_train, y_train)
predictions = knn_model3.predict(x_test)
knn_stats3 = ClassificationStatistics(y_test, predictions)
matrix = knn_stats3.calculate_confusion_matrix()


knn_model4 = KNNClassificationModel(norm="l1",neighbors_n=30)
knn_model4.fit(x_train, y_train)
predictions = knn_model4.predict(x_test)
knn_stats4 = ClassificationStatistics(y_test, predictions)
matrix = knn_stats4.calculate_confusion_matrix()


knn_model5 = KNNClassificationModel(norm="l2",neighbors_n=30)
knn_model5.fit(x_train, y_train)
predictions = knn_model5.predict(x_test)
knn_stats5 = ClassificationStatistics(y_test, predictions)
matrix = knn_stats5.calculate_confusion_matrix()


#Collect statistics
statistics = {
   'L1\nneighbors=3': knn_stats0.calculate_all(),
   'L2\nneighbors=3': knn_stats1.calculate_all(),
   
   'L1\nneighbors=20': knn_stats2.calculate_all(),
   'L2\nneighbors=20': knn_stats3.calculate_all(),
   
   'L1\nneighbors=30': knn_stats4.calculate_all(),
   'L1\nneighbors=30': knn_stats5.calculate_all()
}


#Plot statistics
plot_all_T(list(statistics.values()),list(statistics.keys()),"Compare KNN Classification Models", 2,[0,1.19], 15)
plt.show()