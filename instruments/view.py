
import numpy as np
import matplotlib.pyplot as plt
            
def show_confusion_matrix(matrix: np.ndarray, labels: list) -> None:
    fig, axe = plt.subplots()
    graph = axe.matshow(matrix,cmap = "Blues")
    axe.set_xlabel("Predicted class")
    axe.set_ylabel("Observed class")
    axe.set_xticklabels(labels)
    axe.set_yticklabels(labels)
    fig.colorbar(graph, ax = axe) 