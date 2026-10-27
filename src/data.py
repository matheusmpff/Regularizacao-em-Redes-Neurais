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
    def __init__(self,
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