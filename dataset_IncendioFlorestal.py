import sys
import scipy
import numpy
import matplotlib
import pandas as pd
import sklearn

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


######################Em desenvolvimento###########################



# ler a url. É possível baixar o arquivo *.csv e adicionar o path
url = 'https://corgis-edu.github.io/corgis/datasets/csv/video_games/video_games.csv'
#definir atributos para os nomes das colunas
attributes = ["Title", "Features.Handheld", "Features.Max Players", "Features.Multiplatform", "class"]
#na leitura dos dados, o parâmetro names é usado para definir o nome das colunas
dataset = pd.read_csv(url, names = attributes)
dataset.columns = attributes

#ter uma dimensão de quantas instâncias (linhas) e quantos atributos (colunas) os dados contêm
print(dataset.shape)

#analisar os dados
print(dataset.head(150))

#ver resumo estatístico: contagem, média, valores mínimo e máximo, e alguns percentuais
print(dataset.describe())

#distribuição por classe
print(dataset.groupby("class").size())

#criar gráfico de caixas (bloxplot)
dataset.plot(kind='box', subplots=False, layout=(2,2))
plt.show()

# histogramas - diagramas de uma variável
dataset.hist()
plt.show()

# Gráficos Multivariados (observar presença de agrupamentos diagonais)
scatter_matrix(dataset)
plt.show()

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size)

#treinando o classificador
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)

#fazendo predições
Y_pred = classifier.predict(X_validation)

print(confusion_matrix(Y_validation, Y_pred))
print(classification_report(Y_validation, Y_pred))

Y_pred

print()