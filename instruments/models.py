import scipy
from collections import Counter
import numpy as np
from abc import ABC
from instruments.dftools import *

class ClassificationModel(ABC):
    
    def __init__(self, regularization: str = None, regularization_factor: float = 1):
        
        if(regularization == "l1"):
            self.regularization_function = self.__l1
        elif(regularization == "l2"):
            self.regularization_function = self.__l2 
        else:
            self.regularization_function = lambda _: 0
        self.normals = []
        self.regularization_factor = float(regularization_factor)
            
    def __minimize(self, objects, targets, x0 = None) -> list:
        loss_function = self.loss_function
        args = (objects, targets)
        method="trust-constr"
        tol=10**(-12)
        
        if(x0 == None):
            x0 = np.zeros(len(objects[0]))
        return scipy.optimize.minimize(loss_function, x0 = x0, args = args, method=method, tol=tol).x
    
    def __l1(self, vector: np.ndarray) -> float:      
        _sum = 0
        for i in range(len(vector)-1):
            _sum += abs(vector[i])
        return _sum*self.regularization_factor
    
    def __l2(self, vector: np.ndarray) -> float:     
        _sum = 0
        for i in range(len(vector)-1):
            _sum += vector[i]**2
        return _sum*self.regularization_factor
    
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
    
    def __init__(self, regularization: str = None, regularization_factor: float = 1):
        super().__init__(regularization, regularization_factor)
        self.loss_function = lambda normal, objects, targets: self.regularization_function(normal) + self.__quadric_loss(normal, objects, targets)

    def __quadric_loss(self, normal, objects, targets) -> float:
        _sum = 0
        for object, target in zip(objects,targets):
            margin = target*np.dot(object,normal)
            _sum += (1-margin)**2
        return _sum   
    
class LogisticCModel(ClassificationModel):
    
    def __init__(self, regularization: str = None, regularization_factor: float = 1):
        super().__init__(regularization, regularization_factor)

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
    
    def __init__(self, regularization: str = None, regularization_factor: float = 1, kernel = "linear",d=1,r=0,g=1):
        super().__init__(regularization,regularization_factor)
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


class KNNClassificationModel(ClassificationModel):
    def __init__(self, neighbors_n:int = 3, norm: str = "l2"):
        if(norm == "l1"):
            self.norm = self.__l1
        elif(norm == "l2"):
            self.norm = self.__l2 
        self.neighbors_n = neighbors_n
            
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
    
    def fit(self, objects: np.ndarray, targets: np.ndarray):    
        self.train_objects = objects
        self.train_targets = targets    
            
    def most_frequent(self, arr):
        freq = []
        unique = []
        for x in arr:
            if(x in unique):
                index = unique.index(x)
                freq[index] += 1
            else:
                unique.append(x)
                freq.append(1)
        max_index = freq.index(max(freq))
        return unique[max_index]    
    
    def __get_distances(self, target_point: np.ndarray, points: np.ndarray):
        distances = []
        for point in points:
            distance = self.norm(point-target_point)
            distances.append(distance)
        return distances
    
    def __get_neighbors(self, distances:np.ndarray, targets: np.ndarray):
        pairs = list(zip(distances, targets))
        pairs.sort(key=lambda x: x[0])
        return pairs[:self.neighbors_n]
        
    def predict(self, objects: np.ndarray) -> np.ndarray:
        targets = []
        for obj in objects:
            distances = self.__get_distances(obj, self.train_objects)
            neighbors = self.__get_neighbors(distances, self.train_targets)
            classes = list(map(lambda x: x[1],neighbors) )
            
            target = self.most_frequent(classes)
            targets.append(target)
        return np.array(targets)
    
    
            
                
     
        
   