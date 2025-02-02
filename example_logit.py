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

#Logit model
logit_model = LogisticCModel()
logit_model.fit(x_train,y_train)
logit_predictions = logit_model.predict(x_test)
logit_stats0 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats0.calculate_confusion_matrix()


#Logit model
logit_model = LogisticCModel(regularization="l1",regularization_factor=0.5)#-0.125
logit_model.fit(x_train,y_train)
logit_predictions = logit_model.predict(x_test)
logit_stats1 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats1.calculate_confusion_matrix()

#Logit model
logit_model = LogisticCModel(regularization="l2",regularization_factor=0.5)#0.007
logit_model.fit(x_train,y_train)
logit_predictions = logit_model.predict(x_test)
logit_stats2 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats2.calculate_confusion_matrix()


#Logit model
logit_model = LogisticCModel(regularization="l1",regularization_factor=2)
logit_model.fit(x_train,y_train)
logit_predictions = logit_model.predict(x_test)
logit_stats3 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats3.calculate_confusion_matrix()

#Logit model
logit_model = LogisticCModel(regularization="l2",regularization_factor=2)
logit_model.fit(x_train,y_train)
logit_predictions = logit_model.predict(x_test)
logit_stats4 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats4.calculate_confusion_matrix()


#Logit model
logit_model = LogisticCModel(regularization="l1",regularization_factor=5)
logit_model.fit(x_train,y_train)
logit_predictions = logit_model.predict(x_test)
logit_stats5 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats5.calculate_confusion_matrix()

#Logit model
logit_model = LogisticCModel(regularization="l2",regularization_factor=5)
logit_model.fit(x_train,y_train)
logit_predictions = logit_model.predict(x_test)
logit_stats6 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats6.calculate_confusion_matrix()

statistics = {
   'No regularization': logit_stats0.calculate_all(),
   'L1 0.5': logit_stats1.calculate_all(),
   'L2 0.5': logit_stats2.calculate_all(),
   'L1 2': logit_stats3.calculate_all(),
   'L2 2': logit_stats4.calculate_all(),
   'L1 5': logit_stats5.calculate_all(),
   'L2 5': logit_stats6.calculate_all(),
}



plot_all_T(list(statistics.values()),list(statistics.keys()),"Compare Logistic Classification Models", 2,[0,1.19],margin=20,width=1.5)
plt.show()