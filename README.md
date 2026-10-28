# Regularização L2 em Redes Neurais

Este projeto explora o impacto da **regularização L2 em redes neurais do tipo MLP (Multilayer Perceptron)**, comparando o comportamento de duas redes treinadas para resolver o mesmo problema de classificação binária.

O objetivo é visualizar como a regularização influencia os pesos aprendidos, as fronteiras de decisão e a capacidade de generalização dos modelos.

## Tecnologias utilizadas

- **Python** — implementação do experimento.
- **TensorFlow e Keras** — construção e treinamento das redes neurais.
- **NumPy** — operações numéricas e análise dos parâmetros.
- **Scikit-learn** — geração dos dados sintéticos e divisão entre treino e teste.
- **Matplotlib** — geração dos gráficos.
- **Manim** — criação das animações explicativas.

## Metodologia

### 1. Geração dos dados

Foi utilizado o conjunto sintético `make_moons`, disponibilizado pelo Scikit-learn. Ele contém duas classes com distribuição em formato de duas luas, permitindo estudar um problema de classificação não linear.

Os dados são divididos em conjuntos de treinamento e teste. O primeiro é utilizado para ajustar os parâmetros das redes, enquanto o segundo permite avaliar seu desempenho em exemplos que não participaram do treinamento.

### 2. Construção das redes neurais

Foram implementadas duas redes MLP com a mesma arquitetura:

- Uma camada de entrada com duas características.
- Três camadas ocultas com 64 neurônios cada e função de ativação ReLU.
- Uma camada de saída com um neurônio e função sigmoide.

O treinamento utiliza o otimizador Adam e a função de perda *binary cross-entropy*.

A única diferença planejada entre os modelos é a presença da regularização L2 nas camadas ocultas

Ambos são treinados com os mesmos dados e hiperparâmetros, permitindo analisar o efeito da penalização sobre o aprendizado.

### 3. Como funciona a regularização L2?

A regularização L2, também conhecido com Ridge regression ou Tikhonov Regularization, é uma forma  de reduzir o overfitting dos modelos que criamos. Podemos regularizar modelos lineares restringindo seu pesos, para isso adicionamos um termo de penalização à função de perda original. A penalização incentiva o uso de pesos de menor custo por parte do modelo.

## Análises e visualizações

O projeto utiliza gráficos e animações para tornar os efeitos da regularização mais intuitivos.

### Fronteiras de decisão

As fronteiras de decisão mostram como cada rede divide o espaço das características entre as duas classes.

A comparação permite observar como a regularização influencia o formato dessas fronteiras e se o modelo apresenta uma separação excessivamente complexa dos dados.

### Distribuição dos pesos

O histograma compara a distribuição dos pesos aprendidos pelas duas redes.

Essa visualização ajuda a investigar como a regularização modifica a magnitude e a distribuição dos parâmetros. 

### Métricas de desempenho

A acurácia é avaliada separadamente nos conjuntos de treino e teste.

A comparação permite investigar possíveis sinais de overfitting e verificar se o comportamento observado durante o treinamento se mantém em dados não utilizados no ajuste dos parâmetros.

## Benefícios da regularização L2

A regularização L2 pode trazer diversos benefícios:

- **Redução do overfitting:** desencoraja soluções que dependem excessivamente de pesos de grande magnitude.
- **Controle da complexidade do modelo:** pode favorecer fronteiras de decisão menos sensíveis a pequenas variações nos dados.
- **Melhor generalização:** pode ajudar o modelo a apresentar um desempenho mais consistente em exemplos não vistos.

Esses benefícios dependem dos dados, da arquitetura e da intensidade da regularização. Um coeficiente excessivamente alto pode provocar *underfitting*, fazendo com que a rede não consiga aprender adequadamente os padrões existentes.

Portanto, o objetivo não é simplesmente minimizar os pesos, mas encontrar um equilíbrio entre ajustar os dados e generalizar para novos exemplos.

## Video usado gerando Manim
[▶️ Assistir à animação do projeto](midia/videos/animation/1080p60/Regularizacao.mp4)

## Como executar

### Pré-requisitos

- Python compatível com as versões das bibliotecas utilizadas.
- Dependências do projeto instaladas em um ambiente virtual.

### Instalação

Clone o repositório e entre na pasta do projeto:

```bash
git clone <URL_DO_REPOSITORIO>
cd Regularizacao-em-Redes-Neurais
```

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install tensorflow numpy scikit-learn matplotlib manim pandas seaborn
```

Execute a animação a partir da raiz do projeto.

```bash
python -m manim -pqh src/animation.py Regularizacao
```

Este projeto demonstra, de forma prática e visual, como a regularização L2 pode influenciar o treinamento de redes neurais. Ao comparar fronteiras de decisão, distribuições de pesos e métricas de desempenho, é possível compreender melhor a relação entre ajuste aos dados, complexidade do modelo e generalização.

A regularização não garante, por si só, uma acurácia maior. Seu valor está em ajudar a controlar o aprendizado e reduzir o risco de overfitting, quando aplicada de maneira adequada.
