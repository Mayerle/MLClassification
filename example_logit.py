from instruments.linearmodels import *
from instruments.dftools import *
from instruments.statistic import *
from instruments.view import *

from instruments.datasample import *

x_train, x_test, y_train, y_test = get_data()

normals = []
#Logit model
logit_model = LogisticCModel()
normal = logit_model.fit(x_train,y_train)
normals.append(normal)
logit_predictions = logit_model.predict(x_test)
logit_stats0 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats0.calculate_confusion_matrix()

#Logit model
logit_model = LogisticCModel(regularization="l1",regularization_factor=0.5)#-0.125
normal = logit_model.fit(x_train,y_train)
normals.append(normal)
logit_predictions = logit_model.predict(x_test)
logit_stats1 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats1.calculate_confusion_matrix()

#Logit model
logit_model = LogisticCModel(regularization="l2",regularization_factor=0.5)#0.007
normal = logit_model.fit(x_train,y_train)
normals.append(normal)
logit_predictions = logit_model.predict(x_test)
logit_stats2 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats2.calculate_confusion_matrix()


#Logit model
logit_model = LogisticCModel(regularization="l1",regularization_factor=2)
normal = logit_model.fit(x_train,y_train)
normals.append(normal)
logit_predictions = logit_model.predict(x_test)
logit_stats3 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats3.calculate_confusion_matrix()

#Logit model
logit_model = LogisticCModel(regularization="l2",regularization_factor=2)
normal = logit_model.fit(x_train,y_train)
normals.append(normal)
logit_predictions = logit_model.predict(x_test)
logit_stats4 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats4.calculate_confusion_matrix()


#Logit model
logit_model = LogisticCModel(regularization="l1",regularization_factor=5)
normal = logit_model.fit(x_train,y_train)
normals.append(normal)
logit_predictions = logit_model.predict(x_test)
logit_stats5 = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats5.calculate_confusion_matrix()

#Logit model
logit_model = LogisticCModel(regularization="l2",regularization_factor=5)
normal = logit_model.fit(x_train,y_train)
normals.append(normal)
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
labelz = [
   'No    |',
   'L1 0.5|',
   'L2 0.5|',
   'L1 2  |',
   'L2 2  |',
   'L1 5  |',
   'L2 5  |',
]
for l,n in zip(labelz,normals):
   with np.printoptions(precision=3, suppress=True):
      normal = n[0][:-1]
      print(l,[f"{x:.2f}" for x in normal])

plot_all_T(list(statistics.values()),list(statistics.keys()),"Compare Logistic Classification Models", 2,[0,1.19],margin=20,width=1.5)
plt.show()