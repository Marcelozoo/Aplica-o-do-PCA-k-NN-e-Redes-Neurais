from dataframe import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class Pca:

    def __init__(self, dataframe : DataFrame) -> None:
        self.__dataframe = dataframe

    def normalizar_peso_dados(self) -> None:
        """
        Essa função normaliza os dados para que eles tenham o mesmo grau de importância ou seja
        agora os atributos possuem o mesmo peso.
        """
        scaler_total = StandardScaler()
        X_train_scaled = scaler_total.fit_transform(self.__dataframe.get_dataframe_X())
    
    def descobrindo_variancia_dos_dados(self) -> None:
        """
        Essa função tem como objetivo obter a variancia explicada para cada componente usando
        o método fit que indica as direções das variações dos componente principais.
        """
        self.__pca = PCA(n_components=None)
        self.__pca.fit(self.__dataframe.get_dataframe_X())
        self.__explained_variance = self.__pca.explained_variance_ratio_

    def definindo_numero_de_componentes_com_maior_variancia(self) -> None:
        """
        Essa função acha qual a quantidade de componente principais explicam a maior parte
        da variância dos dados e depois aplica o PCA no conjunto X_train.
        """
        self.__n_components = np.argmax(self.__explained_variance) + 1 
        print("\nNúmero de componentes escolhidos:", self.__n_components)

    def aplica_pca_no_conjunto_x_train(self) -> None:
        """
        Essa função aplica o PCA no conjunto de dados X_train.
        """
        pca_treinamento = PCA(n_components = self.__n_components)
        self.__X_train_pca = self.__pca.fit_transform(self.__dataframe.get_x_train())

    def get_x_train_pca(self) -> pd:
        return self.__X_train_pca
    
    def get_pca(self) -> PCA:
        return self.__pca
    

    

  