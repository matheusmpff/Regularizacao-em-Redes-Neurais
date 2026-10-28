import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


    

class ExperimentData:
    def __init__(self,X,y,X_train,y_train,X_test,y_test):
        #Atributos
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

def create_scatterplot(model, data:ExperimentData, caminho):
    x_min , x_max= -2, 3
    y_min, y_max= -2, 2
    
    xx,yy = np.meshgrid(
        np.linspace(x_min,x_max,200),
        np.linspace(y_min,y_max,200)
    )
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    probability = model.predict(grid,verbose=0).reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    
    ax.contourf(
        xx,
        yy,
        probability,
        levels= np.linspace(0,1,21),
        cmap="coolwarm",
        alpha=0.4,
        vmin=0,vmax=1
    )
    #Deseja a linha em que probabilidade é 50%
    ax.contour(
        xx,
        yy,
        probability,
        levels=[0.5],
        colors="#151818",
        linewidth=2
    )
    
    df_train = pd.DataFrame({
        "x1": data.X_train[:,0],
        "x2": data.X_train[:,1],
        "classe": data.y_train
    })
    
    sns.scatterplot(
        data=df_train,
        x="x1",
        y="x2",
        hue="classe",
        palette={0: "#2474B5", 1: "#E76F51"},
        s=17,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.85,
        ax=ax
    )
    
    plt.savefig(caminho,dpi=300,bbox_inches="tight")
    
class ExperimentResult:
    def __init__(
        self,
        model_without_l2,
        model_with_l2,
        acc_train_without_l2,
        acc_train_with_l2,
        acc_test_without_l2,
        acc_test_with_l2,
        weights_without_l2,
        weights_with_l2
        ):
        
        self.model_without_l2 = model_without_l2
        self.model_with_l2 = model_with_l2
        self.acc_train_without_l2 = acc_train_without_l2
        self.acc_train_with_l2 = acc_train_with_l2
        self.acc_test_without_l2 = acc_test_without_l2
        self.acc_test_with_l2 = acc_test_with_l2
        self.weights_without_l2 = weights_without_l2
        self.weights_with_l2 = weights_with_l2
    
    def create_weights_hist(self):
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
        sns.histplot(
            data=self.weights_with_l2,
            bins=55,
            ax=ax,
            color="#253DA8",
            stat="density",
            alpha=0.6
        )
        sns.histplot(
            data=self.weights_without_l2,
            bins=55,
            ax=ax,
            color="#C02525",
            stat="density",
            alpha=0.6
        )
        plt.xlabel("Valores dos pesos")
        plt.ylabel("Densidade")
        plt.savefig("../media/imgs/comparacao_pesos.png",dpi=300,bbox_inches="tight")
    
    def create_models_scatterplots(self,data:ExperimentData):
        create_scatterplot(self.model_without_l2,data,"../media/imgs/scatterplot.png")
        create_scatterplot(self.model_with_l2,data,"../media/imgs/scatterplot2.png")      