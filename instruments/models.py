import scipy
import numpy as np
from abc import ABC
from instruments.dftools import *

class ClassificationModel(ABC):
    
    def __init__(self, regularization: str = None):
        
        if(regularization == "l1"):
            self.regularization_function = self.__l1
        elif(regularization == "l2"):
            self.regularization_function = self.__l2 
        else:
            self.regularization_function = lambda _: 0
            
    def __minimize(self, objects, targets, x0 = None) -> list:
        loss_function = self.loss_function
        args = (objects, targets)
        method="trust-constr"
        tol=10**(-10)
        
        if(x0 == None):
            x0 = np.zeros(len(objects[0]))
        return scipy.optimize.minimize(loss_function, x0 = x0, args = args, method=method, tol=tol).x
    
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
    

class QuadricCModel(ClassificationModel):
    
    def __init__(self, regularization: str = None):
        super().__init__(regularization)
        self.loss_function = lambda normal, objects, targets: self.regularization_function(normal) + self.__quadric_loss(normal, objects, targets)

    def __quadric_loss(self, normal, objects, targets) -> float:
        _sum = 0
        for object, target in zip(objects,targets):
            margin = target*np.dot(object,normal)
            _sum += (1-margin)**2
        return _sum   
    
class LogisticCModel(ClassificationModel):
    
    def __init__(self, regularization: str = None):
        super().__init__(regularization)

        self.loss_function = lambda normal, objects, targets: self.regularization_function(normal) + self.__logistic_loss(normal, objects, targets)
    def __sigmoid(self, x) -> float:
        return (1+np.exp(-x))**(-1)       
    
    def __logistic_loss(self, normal, objects, targets) -> float:
        _sum = 0
        for object, target in zip(objects,targets):
            margin = target*np.dot(object,normal)
            _sum += -np.log(self.__sigmoid(margin))
        return _sum 
   
class SVMCModel(ClassificationModel):
    
    def __init__(self, regularization: str = None, kernel = "linear",d=1,r=0,g=1):
        super().__init__(regularization)
        self.loss_function = lambda normal, objects, targets: self.regularization_function(normal) + self.__hinge_loss(normal, objects, targets)
        self.kernel = lambda x,y: np.dot(x,y)
        if(kernel == "linear"):
            self.kernel = lambda x,y: np.dot(x,y)
        elif(kernel == "poly"):
            self.kernel = lambda x,y: np.dot(x,y+r)**d
        elif(kernel == "gaussian"):
            self.kernel = lambda x,y: np.exp( -g*np.dot(x-y, x-y) )
        elif(kernel == "sigmoid"):
            self.kernel = lambda x,y: np.tanh( g*np.dot(x,y) + r ) 
            
    def __hinge_loss(self,normal, objects, targets) -> float:
        _sum = 0
        for object, target in zip(objects,targets):
            margin = target*self.kernel(object,normal)
            _sum += max(0,1-margin)
        return _sum
    def predict(self, objects: np.ndarray) -> np.ndarray:
        predictions = np.ndarray(len(objects))
        for i in range(len(objects)):
            projections = []
            for normal in self.normals:
                projections.append(self.kernel(normal, objects[i]))
            best_prediction = np.argmax(projections)
            predictions[i] = best_prediction
        return predictions

