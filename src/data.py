class ExperimentData:
    def __init__(self,data,X_train,y_train,X_test,y_test):
        #Atributos
        self.data = data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

class ExperimentResult:
    def __init__(self,
                    model_without_l2,
                    model_with_l2,
                    accuracy_without_l2,
                    accuracy_with_l2,
                    weights_without_l2,
                    weights_with_l2
                ):
        
        self.model_without_l2 = model_without_l2
        self.model_with_l2 = model_with_l2
        self.accuracy_without_l2 = accuracy_without_l2
        self.accuracy_with_l2 = accuracy_with_l2
        self.weights_without_l2 = weights_without_l2
        self.weights_with_l2 = weights_with_l2