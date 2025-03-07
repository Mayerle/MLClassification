from instruments.linearmodels import *
from instruments.dftools import *
from instruments.statistic import *
from instruments.view import *
from instruments.datasample import *
from instruments.metricmodels import *

x_train, x_test, y_train, y_test = get_data()

#Quadric model
quadric_model = QuadricCModel()
quadric_model.fit(x_train,y_train)
quadric_predictions = quadric_model.predict(x_test)
quadric_stats = ClassificationStatistics(y_test,quadric_predictions)
matrix = quadric_stats.calculate_confusion_matrix()


#Logit model
logit_model = LogisticCModel(regularization="l2",regularization_factor=0.5)
normal = logit_model.fit(x_train,y_train)
logit_predictions = logit_model.predict(x_test)
logit_stats = ClassificationStatistics(y_test,logit_predictions)
matrix = logit_stats.calculate_confusion_matrix()


#SVM model
svm_model = SVMCModel(kernel="gaussian",g = 0.5)
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()


#KNN model
knn_model = KNNClassificationModel(norm="l2",neighbors=20)
knn_model.fit(x_train, y_train)
predictions = knn_model.predict(x_test)
knn_stats = ClassificationStatistics(y_test, predictions)
matrix = knn_stats.calculate_confusion_matrix()

#Collect statistics
statistics = {
    'Quadric\nNo regularization': quadric_stats.calculate_all(),
    'Logit\nRegularization L2 0.5': logit_stats.calculate_all(),
    'SVM\nkernel=gaussian\ng=0.5': svm_stats.calculate_all(),
    'KNN\nNorm=L2\nneighbors=20': knn_stats.calculate_all(),
}

#Plot statistics
plot_statistics(list(statistics.values()),list(statistics.keys()),"Compare Best Classification Models", 2,[0,1.19],margin=20,width=1.5)
plt.show()