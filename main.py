import pandas as pd
from instruments.dftools import *

features_columns = ["SepalLengthCm","SepalWidthCm", "PetalLengthCm",  "PetalWidthCm"]
target_column = "Species"

seed = 1
df = pd.read_csv("dataset/iris.csv").sample(frac=1,random_state=seed)
objects = df[features_columns]
targets = df[target_column]
objects = normalize_features(objects)
objects = convert_to_onevectors(objects)
targets = one_hot_encode(targets)

x_train, x_test, y_train, y_test = split_data(objects,targets)

#print(one_hot_encode(y_train))

class ClassificationModel:
    
    def __init__(self, model_type: str, regularization: str = None):
        
        if(regularization == "l1"):
            self.regularization_function = self.__l1
        elif(regularization == "l2"):
            self.regularization_function = self.__l2 
        else:
            self.regularization_function = lambda _: 0
        
        if(model_type == "quadric"):
            self.loss_function = lambda normal, objects, targets: self.regularization_function(normal) + self.__quadric_loss(normal, objects, targets)
    
    def __minimize(self, objects, targets, x0 = None) -> list:
        loss_function = self.loss_function
        args = (objects, targets)
        method="trust-constr"
        tol=10**(-10)
        
        if(x0 == None):
            x0 = np.zeros(len(objects[0]))
        print(self.regularization_function(x0))
        
        print(loss_function(x0,objects,objects))
        return scipy.optimize.minimize(loss_function, x0 = x0, args = args, method=method, tol=tol).x
    
    def __quadric_loss(self, normal, objects, targets) -> float:
        _sum = 0
        for object, target in zip(objects,targets):
            _sum += (1-target*np.dot(object,normal))**2
        return _sum    
            
    def __l1(self, vector: np.ndarray) -> float:      
        _sum = 0
        for i in range(len(vector)-1):
            _sum += abs(vector[i])
        return _sum
    
    def __l2(self, vector: np.ndarray) -> float:     
        _sum = 0
        for i in range(len(vector)-1):
            _sum += vector[i]**2
        return _sum
    
    def __fit_class(self, target_class, objects: np.ndarray, targets: np.ndarray, x0 = None):
        
        targets = label_encode(targets, target_class)
        return self.__minimize(objects,targets,x0)
    
    def fit(self, objects: np.ndarray, targets: np.ndarray, x0: np.ndarray = None):    
            
        uniques_classes = np.unique(targets)
        self.normals = []
        for target_class in uniques_classes:
            self.normals.append(self.__fit_class(target_class,objects,targets, x0)) 
        return self.normals   
    def predict(self, objects: np.ndarray) -> np.ndarray:
        predictions = np.ndarray(len(objects))
        for i in range(len(objects)):
            projections = []
            for normal in self.normals:
                projections.append(np.dot(normal, objects[i]))
            best_prediction = np.argmax(projections)
            predictions[i] = best_prediction
        return predictions
#print(np.array(x_train).shape)
#print(x_train[0].shape[0])
#print(x_train, y_train)

model = ClassificationModel("quadric")
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(list(zip(predictions,y_test)))