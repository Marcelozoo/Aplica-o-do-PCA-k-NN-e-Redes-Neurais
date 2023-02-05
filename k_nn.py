#BIBLIOTECAS USADAS: SKLEARN , ARQUIVO MAIN.PY E O ARQUIVO PCA.PY ESTA SENDO IMPORTADO POIS SUAS VARIAVEIS
#SERAO UTILIZADAS NO CÓDIGO ABAIXO
#PARA BAIXAR AS BIBLIOTECAS USE:  pip install sklearn 
#TRABALHO FEITO POR MARCELO BENTO CÔGO E ARTHUR SANTOS ALMEIDA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import PCA
import main


#AQUI ESTOU CRIANDO AS VARIAVEIS PARA TESTAR O K-NN,SAO ELAS: NUMERO DE VIZINHOS = n_neighbors_values
# weihts_values = peso dos vizinhos, que pode ser uniform ou seja mesmo peso para todos ou distance que os vizinhos mais proximos tem maior peso.
#e por ultimo p_values que calculaa distancia e quando seu valor é 1 estou usando distancia de Mahattan quando o p = 2 estou usando a distancia Euclidiana.
#Nota importante: é possivel alterar os valores n_neighbors_values ou colocar mais numeros para teste.

n_neighbors_values = [1,3,5,7,9]
weights_values = ['uniform', 'distance']
p_values = [1, 2]

#Aqui é criado variaveis para armazenar os melhores parametors e a melhor pontuação
best_accuracy = 0
print("\n")
#Aqui é um for para ir testando os parametros usando diversas combinacoes que foram criadas acima para ver qual possui melhor pontuação
for n_neighbors in n_neighbors_values:
    for weights in weights_values:
        for p in p_values:
            #Aqui criamos o modelo KNeighborsClassifier com os parâmetros atuais
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
            #Aqui treinamos o modelo com o conjunto de treinamento
            knn.fit(PCA.X_train_pca, main.Y_train.values.ravel())
            #Aqui fazemos as previsões com o conjunto de validação
            y_pred = knn.predict(PCA.pca.transform(main.X_valid))
            #Agora calculamos a precisão do modelo com o conjunto de validação
            accuracy = accuracy_score(main.Y_valid,y_pred)
            print("Os parametros usados: vizinhos: {}, W:{} e p {} tiveram {}".format(n_neighbors,weights,p,accuracy))
            #e por ultimo verificamos se esse é o melhor desempenho até agora
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (n_neighbors, weights, p)

# Imprime os melhores parâmetros encontrados
# o * está apenas desempacotando a lista para ser usando na funcao .format.
print("\nMelhores parâmetros: n_neighbors={}, weights={}, p={}".format(*best_params))


#Cria o modelo KNeighborsClassifier com os melhores parâmetros encontrados
knn = KNeighborsClassifier(n_neighbors=best_params[0], weights=best_params[1], p=best_params[2])
# Treina o modelo
knn.fit(PCA.X_train_pca, main.Y_train.values.ravel())

y_test_pred = knn.predict(PCA.pca.transform(main.X_test))
#Agora calcula a accuracy com o medelo de teste
test_accuracy = accuracy_score(main.Y_test, y_test_pred)
print("\nPrecisão do modelo no conjunto de teste: {:.2f}%".format(test_accuracy * 100))


