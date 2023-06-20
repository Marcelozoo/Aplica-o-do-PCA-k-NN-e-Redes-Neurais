from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from dataframe import DataFrame
from pca import Pca

class Knn:

    def __init__(self, dataframe: DataFrame, pca : Pca) -> None:
        """
        Esse construtor cria as variáveis para testar o K-NN, são elas:
        self.__n_neighbors_values = numero de vizinhos
        self.__weights_values = peso dos vizinhos, que pode ser uniform ou seja mesmo peso para todos ou 
            distance que os vizinhos mais proximos tem maior peso.
        self.__p_values = que calculaa distancia e quando seu valor é 1 estou usando distancia de Mahattan quando 
            o p = 2 estou usando a distancia Euclidiana.
        """
        self.__n_neighbors_values = [1,3,5,7,9]
        self.__weights_values = ['uniform', 'distance']
        self.__p_values = [1, 2]
        self.__dataframe = dataframe
        self.__pca = pca

    def descobrir_melhor_combinacao_parametros(self) -> None:
        """
        Essa função  testa diversos parâmetros para obter a combinação com a melhor precisão.
        """
        best_accuracy = 0
        print("\n\tREALIZANDO AS COMBINAÇÕES PARA DESCOBRIR OS MELHORES PARÂMETROS")
        for n_neighbors in self.__n_neighbors_values:
            for weights in self.__weights_values:
                for p in self.__p_values:
                    #Aqui criamos o modelo KNeighborsClassifier com os parâmetros atuais
                    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
                    #Aqui treinamos o modelo com o conjunto de treinamento
                    knn.fit(self.__pca.get_x_train_pca(), self.__dataframe.get_y_train().values.ravel())
                    #Aqui fazemos as previsões com o conjunto de validação
                    y_pred = knn.predict(self.__pca.get_pca().transform(self.__dataframe.get_x_valid()))
                    #Agora calculamos a precisão do modelo com o conjunto de validação
                    accuracy = accuracy_score(self.__dataframe.get_y_valid(), y_pred)
                    print("Vizinhos: {:<7} Weights: {:10} P: {:<10} precisão: {:<7}".format(n_neighbors, weights, p, accuracy))
                    #e por ultimo verificamos se esse é o melhor desempenho até agora
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        self.__best_params = (n_neighbors, weights, p)

    def imprimir_melhores_parametros(self) -> None:
        """
        Essa função imprimi os parâmetros com a melhor precisão.
        """
        print("\nMelhores parâmetros: n_neighbors = {}, weights = {}, p = {}".format(*self.__best_params))

    
    def cria_modelo_com_os_melhores_parametros(self) -> None:
        """
        Essa função cria o modelo KNeighborsClassifier com os melhores parâmetros encontrados.
        """
        self.__knn = KNeighborsClassifier(
            n_neighbors=self.__best_params[0], weights=self.__best_params[1], p=self.__best_params[2]
            )

    def treinar_modelo(self) -> None:
        """
        Essa função treina o modelo.
        """
        self.__knn.fit(self.__pca.get_x_train_pca(), self.__dataframe.get_y_train().values.ravel())
        
    def realiza_previsoes(self) -> None:
        """
        Essa função calcula a previsão com os  dados de  X_test.
        """
        self.__y_test_pred = self.__knn.predict(self.__pca.get_pca().transform(self.__dataframe.get_x_test()))

    def calcular_precisao_modelo(self) -> None:
        """
        Essa função calcula a precisão com os dados de Y_test.
        """
        test_accuracy = accuracy_score(self.__dataframe.get_y_test(), self.__y_test_pred)
        print("\nPrecisão do modelo no conjunto de teste: {:.2f}%".format(test_accuracy * 100))