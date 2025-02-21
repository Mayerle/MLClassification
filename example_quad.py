from instruments.linearmodels import *
from instruments.dftools import *
from instruments.statistic import *
from instruments.view import *

from instruments.datasample import *

x_train, x_test, y_train, y_test = get_data()

#Quadric model
quad_model = QuadricCModel()
quad_model.fit(x_train,y_train)
quad_predictions = quad_model.predict(x_test)
quad_stats0 = ClassificationStatistics(y_test,quad_predictions)
matrix = quad_stats0.calculate_confusion_matrix()


#Quadric model 0.5L1
quad_model = QuadricCModel(regularization="l1",regularization_factor=0.5)#-0.125
quad_model.fit(x_train,y_train)
quad_predictions = quad_model.predict(x_test)
quad_stats1 = ClassificationStatistics(y_test,quad_predictions)
matrix = quad_stats1.calculate_confusion_matrix()

#Quadric model 0.5L2
quad_model = QuadricCModel(regularization="l2",regularization_factor=0.5)#0.007
quad_model.fit(x_train,y_train)
quad_predictions = quad_model.predict(x_test)
quad_stats2 = ClassificationStatistics(y_test,quad_predictions)
matrix = quad_stats2.calculate_confusion_matrix()


#Quadric model 2L1
quad_model = QuadricCModel(regularization="l1",regularization_factor=2)
quad_model.fit(x_train,y_train)
quad_predictions = quad_model.predict(x_test)
quad_stats3 = ClassificationStatistics(y_test,quad_predictions)
matrix = quad_stats3.calculate_confusion_matrix()

#Quadric model 2L2
quad_model = QuadricCModel(regularization="l2",regularization_factor=2)
quad_model.fit(x_train,y_train)
quad_predictions = quad_model.predict(x_test)
quad_stats4 = ClassificationStatistics(y_test,quad_predictions)
matrix = quad_stats4.calculate_confusion_matrix()


#Quadric model 5L1
quad_model = QuadricCModel(regularization="l1",regularization_factor=5)
quad_model.fit(x_train,y_train)
quad_predictions = quad_model.predict(x_test)
quad_stats5 = ClassificationStatistics(y_test,quad_predictions)
matrix = quad_stats5.calculate_confusion_matrix()

#Quadric model 5L2
quad_model = QuadricCModel(regularization="l2",regularization_factor=5)
quad_model.fit(x_train,y_train)
quad_predictions = quad_model.predict(x_test)
quad_stats6 = ClassificationStatistics(y_test,quad_predictions)
matrix = quad_stats6.calculate_confusion_matrix()

statistics = {
   'No regularization': quad_stats0.calculate_all(),
   'L1 0.5': quad_stats1.calculate_all(),
   'L2 0.5': quad_stats2.calculate_all(),
   'L1 2': quad_stats3.calculate_all(),
   'L2 2': quad_stats4.calculate_all(),
   'L1 5': quad_stats5.calculate_all(),
   'L2 5': quad_stats6.calculate_all(),
}



plot_all_T(list(statistics.values()),list(statistics.keys()),"Compare Quadric Classification Models", 2,[0,1.19],margin=20,width=1.5)
plt.show()