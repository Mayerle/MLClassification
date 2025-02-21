import numpy as np
from instruments.dftools import *
class KNNClassificationModel:
    def __init__(self, neighbors:int = 3, metric: str = "l2",**kwargs):
        """
        :param int neighbors: Count of nearest neighbors
        :param str metric: "l1", "l2" or "minkowski"\n
                          if metric is "minkowski" kward "minkowski_power" is required
        """
        if(metric == "l1"):
            self.metric = self._l1
        elif(metric == "l2"):
            self.metric = self._l2 
        elif(metric=="minkowski"):
            self.metric = lambda point0,point1: self._minkowski_metric(point0,point1, kwargs["minkowski_power"])
        elif(metric=="cosine"):
            self.metric = self._cosine
        self.neighbors = neighbors
        
    def _l1(self, point0: np.ndarray, point1: np.ndarray) -> float:  
        vector = point1-point0    
        _sum = 0
        for i in range(len(vector)-1):
            _sum += abs(vector[i])
        return _sum
    
    def _l2(self, point0: np.ndarray, point1: np.ndarray) -> float:     
        _sum = 0
        vector = point1-point0
        for i in range(len(vector)-1):
            _sum += vector[i]**2
        return _sum
    
    def _cosine(self, point0: np.ndarray, point1: np.ndarray) -> float:     
        distance0 = self._l2(point0,np.zeros(point0.shape[0]))
        distance1 = self._l2(point1,np.zeros(point1.shape[0]))
        return np.dot(point0[:-1],point1[:-1])/(distance0*distance1)
    
    def _minkowski_metric(self, point0: np.ndarray, point1: np.ndarray, p: float)-> float: 
        _sum = 0
        vector = point1-point0
        for i in range(len(vector)-1):
            _sum += np.power(abs(vector[i]),p)
        return np.power(_sum,1/p)
    
    def _get_distances(self, target: np.ndarray, points: np.ndarray):
        distances = np.ndarray(points.shape[0])
        i = 0
        while (i<points.shape[0]):
            distances[i] = self.metric(target,points[i])
            i+=1
        return distances
    
    def fit(self, objects: np.ndarray, targets: np.ndarray):    
        self.train_objects = objects
        self.train_targets = targets    
            
    def _get_most_frequent(self, arr):
        frequencies = []
        uniques = []
        for x in arr:
            if(x in uniques):
                index = uniques.index(x)
                frequencies[index] += 1
            else:
                uniques.append(x)
                frequencies.append(1)
        max_index = frequencies.index(max(frequencies))
        return uniques[max_index]    
    
    
    
    def _select_class(self,neighbors_targets):
        targets = list(map(lambda x: x[1], neighbors_targets))
        prediction = self._get_most_frequent(targets)
        return prediction    
    
    def predict(self, objects: np.ndarray) -> np.ndarray:
        predictions = np.ndarray(objects.shape[0])
        i = 0
        while (i<objects.shape[0]):
            target_object = objects[i]
            distances = self._get_distances(target_object, self.train_objects)
            pairs = list(zip(distances, self.train_targets))
            pairs.sort(key=lambda x: x[0])
            pairs = pairs[:self.neighbors]
            predictions[i] = self._select_class(pairs)
            i+=1
        return predictions
          
class WeightedKNNCModel(KNNClassificationModel):
    def __init__(self, neighbors = 3, metric = "l2",weights_type = "linear",**kwargs):
        """
        :param int neighbors: Count of nearest neighbors
        :param str metric: "l1", "l2" or "minkowski"\n
                          if metric is "minkowski" kward "minkowski_power" is required
        :param str weights_type: "linear" or "exp"  
                          if weights_type is "exp" kward "weight_power" is required
        """
        super().__init__(neighbors,metric,**kwargs)
        if(weights_type == "linear"):
            self.weigher = lambda i: i+1
        elif(weights_type == "exp"):
            self.weigher = lambda i: np.power(kwargs["weight_power"],i)
            
    def predict(self, objects: np.ndarray) -> np.ndarray:
        predictions = np.ndarray(objects.shape[0])
        i = 0
        while (i<objects.shape[0]):
            target_object = objects[i]
            distances = self._get_distances(target_object, self.train_objects)
            pairs = list(zip(distances, self.train_targets))
            pairs.sort(key=lambda x: x[0])
            pairs = pairs[:self.neighbors]
            unique_targets = []
            weights = []
            for distance, target in pairs:
                if(target in unique_targets):
                    weights[unique_targets.index(target)] = self.weigher(i)
                else:
                    weights.append(self.weigher(i))
                    unique_targets.append(target)
                    
            predictions[i] = unique_targets[weights.index(max(weights))]
            
            i+=1
        return predictions   

class KernelKNNCModel(KNNClassificationModel):
    def __init__(self, neighbors = 3, metric = "l2",kernel = "rect",h_window = 1,**kwargs):
        """
        :param int neighbors: Count of nearest neighbors
        :param str metric: "l1", "l2" or "minkowski"\n
                          if metric is "minkowski" kward "minkowski_power" is required      
        :param str kernel: "rect", "triangular", "epanechnikov", "biquadrate" or "gaussian"     
        :param float h_window: [0, inf)                  
        """
        super().__init__(neighbors,metric,**kwargs)
        if(kernel == "rect"):
            self.kernel = lambda x: 0.5 if (abs(x) <= 1) else 0
        elif(kernel == "triangular"):
            self.kernel = lambda x: (1-abs(x)) if (abs(x) <= 1) else 0
        elif(kernel == "epanechnikov"):
            self.kernel = lambda x: 0.75*(1-x*x) if (abs(x) <= 1) else 0
        elif(kernel == "biquadrate"):
            self.kernel = lambda x: 0.9375*(1-x*x)*(1-x*x) if (abs(x) <= 1) else 0
        elif(kernel == "gaussian"):
            self.kernel = lambda x: np.exp(-2*x*x)/np.sqrt(2*np.pi)
        self.h_window = h_window
        
    def predict(self, objects: np.ndarray) -> np.ndarray:
        predictions = np.ndarray(objects.shape[0])
        i = 0
        while (i<objects.shape[0]):
            target_object = objects[i]
            distances = self._get_distances(target_object, self.train_objects)
            pairs = list(zip(distances, self.train_targets))
            pairs.sort(key=lambda x: x[0])
            pairs = pairs[:self.neighbors]
            unique_targets = []
            weights = []
            for distance, target in pairs:
                if(target in unique_targets):
                    weights[unique_targets.index(target)] = self.kernel(distance/self.h_window)
                else:
                    weights.append(self.kernel(distance/self.h_window))
                    unique_targets.append(target)
                    
            predictions[i] = unique_targets[weights.index(max(weights))]
            
            i+=1
        return predictions
    
class KDNode:
    def __init__(self,objects,targets,axe = 0,max_axe = 0):
        self.axe = axe
        self.max_axe = max_axe
        self.leafs = [[objects,targets]]
        self.value = None
    def split(self):
        self.axe +=1 
        if(self.axe > self.max_axe):
            self.axe = 0
        for leaf in self.leafs:
            if  (type(leaf) == KDNode):
                leaf.split()
            else:
                objects,targets = leaf
                middle = sum(objects)/objects.shape[0]
                self.value = middle[self.axe]
                condition_left = objects[:,self.axe] < middle[self.axe]
                condition_right = objects[:,self.axe] >= middle[self.axe]
                objects_left = objects[condition_left]
                targets_left = targets[condition_left]
                objects_right = objects[condition_right]
                targets_right = targets[condition_right]
                left  = KDNode(objects_left, targets_left, self.axe,self.max_axe)
                right = KDNode(objects_right,targets_right,self.axe,self.max_axe)
                self.leafs = [left, right]
                return
    def is_terminal(self) -> bool:
        return type(self.leafs[0]) == list
    def get_splits(self):
        values = []
        if(self.value!=None):
            values.append(self.value)
        for leaf in self.leafs:
            if  (type(leaf) == KDNode):
                values.append(leaf.get_splits())
        return values
class KDTreeClassificationModel:
    def __init__(self, neighbors:int = 3, depth:int = 3, metric: str = "l2",**kwargs):
        """
        :param int neighbors: Count of nearest neighbors
        :param int nodes: Count of tree nodes
        :param str metric: "l1", "l2" or "minkowski"\n
                          if metric is "minkowski" kward "minkowski_power" is required
        """
        if(metric == "l1"):
            self.metric = self._l1
        elif(metric == "l2"):
            self.metric = self._l2 
        elif(metric=="minkowski"):
            self.metric = lambda point0,point1: self._minkowski_metric(point0,point1, kwargs["minkowski_power"])
        elif(metric=="cosine"):
            self.metric = self._cosine
        self.neighbors = neighbors
        self.depth = depth

    def _l1(self, point0: np.ndarray, point1: np.ndarray) -> float:  
        vector = point1-point0    
        _sum = 0
        for i in range(len(vector)-1):
            _sum += abs(vector[i])
        return _sum
    
    def _l2(self, point0: np.ndarray, point1: np.ndarray) -> float:     
        _sum = 0
        vector = point1-point0
        for i in range(len(vector)-1):
            _sum += vector[i]**2
        return _sum
    
    def _cosine(self, point0: np.ndarray, point1: np.ndarray) -> float:     
        distance0 = self._l2(point0,np.zeros(point0.shape[0]))
        distance1 = self._l2(point1,np.zeros(point1.shape[0]))
        return np.dot(point0[:-1],point1[:-1])/(distance0*distance1)
    
    def _minkowski_metric(self, point0: np.ndarray, point1: np.ndarray, p: float)-> float: 
        _sum = 0
        vector = point1-point0
        for i in range(len(vector)-1):
            _sum += np.power(abs(vector[i]),p)
        return np.power(_sum,1/p)
    
    def _get_distances(self, target: np.ndarray, points: np.ndarray):
        distances = np.ndarray(points.shape[0])
        i = 0
        while (i<points.shape[0]):
            distances[i] = self.metric(target,points[i])
            i+=1
        return distances
    
    def fit(self, objects: np.ndarray, targets: np.ndarray):
        max_axe = objects.shape[1]-1
        print(max_axe)
        axe = 0
        i = 0
        self.tree = KDNode(objects,targets,axe,max_axe)
        
        while(i < self.depth):
            self.tree.split()
            i+=1
            
    def _get_most_frequent(self, arr):
        frequencies = []
        uniques = []
        for x in arr:
            if(x in uniques):
                index = uniques.index(x)
                frequencies[index] += 1
            else:
                uniques.append(x)
                frequencies.append(1)
        max_index = frequencies.index(max(frequencies))
        return uniques[max_index]    
    
    
    
    def _select_class(self,neighbors_targets):
        targets = list(map(lambda x: x[1], neighbors_targets))
        prediction = self._get_most_frequent(targets)
        return prediction    
    
    def predict(self, objects: np.ndarray) -> np.ndarray:
        #TODO
        pass
        # predictions = np.ndarray(objects.shape[0])
        # i = 0
        # while (i<objects.shape[0]):
        #     target_object = objects[i]
        #     node = self.tree
        #     nodes = []
        #     while(node.is_terminal() == False):
        #         nodes.append(node)
        #         if(target_object[node.axe] < node.value):
        #             node = node.leafs[0]
        #         else:
        #             node = node.leafs[1]
            
            
            
        #     distances = self._get_distances(target_object, self.train_objects)
        #     pairs = list(zip(distances, self.train_targets))
        #     pairs.sort(key=lambda x: x[0])
        #     pairs = pairs[:self.neighbors]
        #     predictions[i] = self._select_class(pairs)
        #     i+=1
        # return predictions
    def get_splits(self):
        return self.tree.get_splits()