from instruments.linearmodels import *
from instruments.dftools import *
from instruments.statistic import *
from instruments.view import *
from instruments.datasample import *

x_train, x_test, y_train, y_test = get_data()
statistics = {}

#SVM model
svm_model = SVMCModel()
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()
statistics.update({"kernel=linear":svm_stats.calculate_all()})

#SVM model poly with d=2 r=0.2
svm_model = SVMCModel(kernel="poly",d=2,r=0.2)
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()
statistics.update({"kernel=poly\nd=2 r=0.2":svm_stats.calculate_all()})

#SVM model poly with g=0.5
svm_model = SVMCModel(kernel="gaussian",g = 0.5)
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()
statistics.update({"kernel=gaussian\ng=0.5":svm_stats.calculate_all()})

#SVM model sigmoid with g=1 r=1.5
svm_model = SVMCModel(kernel="sigmoid",g = 1,r=1.5)
svm_model.fit(x_train,y_train)
svm_predictions = svm_model.predict(x_test)
svm_stats = ClassificationStatistics(y_test,svm_predictions)
matrix = svm_stats.calculate_confusion_matrix()
statistics.update({"kernel=sigmoid\ng=1 r=1.5":svm_stats.calculate_all()})

#Plot statistics
plot_statistics(list(statistics.values()),list(statistics.keys()),"Compare SVM Classification Models", 2,[0,1.19],margin=20,width=1.5)
plt.show()