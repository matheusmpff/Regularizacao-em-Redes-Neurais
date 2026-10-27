from matplotlib import pyplot as plt
import seaborn as sns

class ExperimentData:
    def __init__(self,X,y,X_train,y_train,X_test,y_test):
        #Atributos
        self.X = X,
        self.y = y,
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

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