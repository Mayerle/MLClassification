
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
            
def show_confusion_matrix(matrix: np.ndarray, labels: list) -> None:
    fig, axe = plt.subplots()
    graph = axe.matshow(matrix,cmap = "Blues")
    axe.set_xlabel("Predicted class")
    axe.set_ylabel("Observed class")
    axe.set_xticklabels(labels)
    axe.set_yticklabels(labels)
    fig.colorbar(graph, ax = axe) 
    
def print_all_per_class(data: list, labels: list) -> None:
    for dat, label in zip(data,labels):
        print(label)
        print(f"Precision: {dat[0]:.2f}")
        print(f"Recall   : {dat[1]:.2f}")
        print("\n")
        
def print_all(data: list, label: str) -> None:
        print(label)
        print(f"Accuracy : {data[0]:.2f}")
        print(f"Precision: {data[1]:.2f}")
        print(f"Recall   : {data[2]:.2f}")
        print("\n")

def plot_all(statistics: dict, digits: int) -> list:
    metrics = ("Accuracy", "Precision", "Recall")
    statistics_rounded = {}
    for m, y in statistics.items():
        statistics_rounded.update({m:[round(k,digits) for k in y]}) 
        
    x = np.arange(len(metrics))  # the label locations
    width = 0.15 
    multiplier = 0

    fig, axe = plt.subplots(layout='constrained')

    for model, value in statistics_rounded.items():
        offset = width * multiplier
        rects = axe.bar(x + offset, value, width, label=model)
        axe.bar_label(rects, padding=3)
        multiplier += 1

    axe.set_ylabel('Metric')
    axe.set_title('Model comparison')
    axe.set_xticks(x + width, metrics)
    axe.legend(loc='upper left', ncols=len(metrics))
    axe.set_ylim(0.8, 1)   
    return (fig,axe)

def plot_all_T(statistics: list, bars: list,title: str = "", digits: int= 2, ylim: list = [0.8,1],margin = 5,width =1) -> list:   
    fig, axe = plt.subplots(layout='constrained')
    statistics = [[round(x,digits) for x in y]  for y in statistics]
    
    padding = 0.2
    labels = ["Accuracy","Precision","Recall"]
    colors = ["#D6C6AD","#D20222","#45666B"]

    count = len(statistics)
    base_space = np.linspace(-1,1,count)
    for i in range(count):
        base = base_space[i]
        apr = statistics[i]
        position = (width+padding)*np.linspace(-1,1,3) + margin*base*np.ones(3)
        rects = axe.bar(position, apr, width, label=labels,color=colors)
        axe.bar_label(rects, padding=3)


    axe.set_xticks(margin*base_space, bars)
    axe.set_ylim(*ylim) 
    axe.legend(loc='upper left', ncols=3,labels=labels)
    factor = 0.8
    fig.set_size_inches(16*factor, 9*factor, forward=True)
    axe.set_title(title)
    