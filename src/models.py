import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from data import ExperimentData, ExperimentResult

RANDOM_STATE = 42
EPOCHS = 300

np.random.seed(RANDOM_STATE)
tf.keras.utils.set_random_seed(RANDOM_STATE)
tf.config.experimental.enable_op_determinism()

def generate_data():
    """
    Gera dados artificiais para serem usados no treinamento e no teste
    das MLP's
    """
    
    X, y = make_moons(
        n_samples=500,
        noise=0.35,
        random_state=RANDOM_STATE
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=RANDOM_STATE)
    
    return ExperimentData(X,y,X_train,y_train,X_test,y_test)

def create_model(l2 = 0 ):
    """
    Cria os modelos com 3 camadas ocultas de 64 neurônios usando ReLU para
    ativação e uma camada de saída com 1 neurônio e que usa a função sigmoid
    para ativação.
    
    Caso não se passe um fator para a regularização L2 o modelo é criado sem
    regularização.
    """
    reg = None
    if l2 !=0:
        reg = regularizers.l2(l2)
    
    model = keras.models.Sequential([
        keras.Input(shape= (2,)),
        layers.Dense(64,activation="relu",kernel_regularizer=reg),
        layers.Dense(64,activation="relu",kernel_regularizer=reg),
        layers.Dense(64,activation="relu",kernel_regularizer=reg),
        layers.Dense(1,activation="sigmoid")
        ])

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate= 0.001),
        loss = "binary_crossentropy",
        metrics= ["accuracy"]
    )
    
    return model

def run_models():
    """
    Função responsável por criar os modelos e os dados que serão utilizados
    e treinar os modelos com eles. A função coleta certos valores durante a 
    execução como:
    
    Acurácia de treino e de teste para cada modelos,
    Pesos que os modelos chegaram ao final do treinamento,
    Os modelos finais,
    Os dados artificiais gerados.
    
    Ao final da execução cria os gráficos necessários e retorna os dados 
    coletados por meio de dois objetoscriados a partir das classes descritas 
    no arquivo data.py 
    """
    
    data = generate_data()
    
    # Criação dos modelos
    model_without_l2 = create_model()
    model_with_l2 = create_model(0.02)
    # Treino dos modelos
    model_without_l2.fit(
        data.X_train,
        data.y_train,
        epochs=EPOCHS,
        batch_size=32,
        shuffle=True,
        verbose=0
    )
    model_with_l2.fit(
        data.X_train,
        data.y_train,
        epochs=EPOCHS,
        batch_size=32,
        shuffle=True,
        verbose=0
    )
    # Acurácia para treino e teste dos modelos
    _ , acc_train_without_l2 = model_without_l2.evaluate(data.X_train,data.y_train,verbose=0)
    _ , acc_train_with_l2 =  model_with_l2.evaluate(data.X_train,data.y_train,verbose=0)
    
    _ , acc_test_without_l2 = model_without_l2.evaluate(data.X_test,data.y_test,verbose=0)
    _ , acc_test_with_l2 =  model_with_l2.evaluate(data.X_test,data.y_test,verbose=0)
    
    #Coloca todos os pesos dentro de um único array para pegar os maiores
    weights_without_l2 = np.concatenate([
        layer.get_weights()[0].flatten()
        for layer in model_without_l2.layers
        if len(layer.get_weights()) > 0
    ])
    
    weights_with_l2 = np.concatenate([
        layer.get_weights()[0].flatten()
        for layer in model_with_l2.layers
        if len(layer.get_weights()) > 0
    ])
    
    # Guarda os resultados no objeto
    results = ExperimentResult(
        model_without_l2,
        model_with_l2,
        acc_train_without_l2,
        acc_train_with_l2,
        acc_test_without_l2,
        acc_test_with_l2,
        weights_without_l2,
        weights_with_l2
    )
    
    #Gera os gráficos necessários para a animação
    results.create_models_scatterplots(data)
    results.create_weights_hist()
    
    return results,data