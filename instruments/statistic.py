import numpy as np

class ClassificationStatistics:
    def __init__(self, observations: np.ndarray, predictions: np.ndarray):
        self.observations = observations
        self.predictions = predictions
        self.confusion_matrix = None
        
    def __check_matrix(self) -> None:
        if self.confusion_matrix is None:
            raise TypeError("Confusion matrix have not calculated!")
        
    def calculate_confusion_matrix(self) -> np.ndarray:
        class_count = len(np.unique(self.observations))
        confusion_matrix = np.zeros((class_count, class_count))
        for observation, prediction in zip(self.observations,self.predictions):
            confusion_matrix[int(observation), int(prediction)] += 1
        self.confusion_matrix = confusion_matrix
        return confusion_matrix

    def calculate_accuracy(self) -> float:
        self.__check_matrix()
        diagonal = self.confusion_matrix.diagonal()
        all_count = self.confusion_matrix.sum()
        correct = diagonal.sum()
        return float(correct/all_count)
    
    def calculate_precisions(self) -> np.ndarray:
        self.__check_matrix()
        diagonal = self.confusion_matrix.diagonal()    
        all_count = self.confusion_matrix.sum(0)
        return diagonal/all_count
    
    def calculate_recalls(self) -> np.ndarray:
        self.__check_matrix()
        diagonal = self.confusion_matrix.diagonal()    
        all_count = self.confusion_matrix.T.sum(0)
        return diagonal/all_count
    
    def calculate_all(self) -> list:
        accuracy   = self.calculate_accuracy()
        precisions = self.calculate_precisions()
        recalls    = self.calculate_recalls()
        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        return [accuracy,precision, recall]
    
    def calculate_f_score(self, b: float = 1) -> float:
        precisions = self.calculate_precisions()
        recalls    = self.calculate_recalls()
        precision = float(np.mean(precisions))
        recall = float(np.mean(recalls))
        score = (1+b**2)*precision*recall/((b**2)*precision+recall)
        return score
    
    def calculate_all_per_class(self) -> list:
        precisions = self.calculate_precisions()
        recalls    = self.calculate_recalls()
        return list(zip(precisions, recalls)) 