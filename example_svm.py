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

statistics = {}

#SVM model
svm_model = SVMCModel()
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()
statistics.update({"kernel=linear":svm_stats.calculate_all()})

#SVM model
svm_model = SVMCModel(kernel="poly",d=2,r=0.2)
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()
statistics.update({"kernel=poly\nd=2 r=0.2":svm_stats.calculate_all()})


#SVM model
svm_model = SVMCModel(kernel="gaussian",g = 0.5)
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()
statistics.update({"kernel=gaussian\ng=0.5":svm_stats.calculate_all()})


#SVM model
svm_model = SVMCModel(kernel="sigmoid",g = 1,r=1.5)
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()
statistics.update({"kernel=sigmoid\ng=1 r=1.5":svm_stats.calculate_all()})




plot_all_T(list(statistics.values()),list(statistics.keys()),"Compare SVM Classification Models", 2,[0,1.19],margin=20,width=1.5)
plt.show()