# A PASTA Rainsin_Dataset é o arquivo baixado do site e dentro dele existe uma planinha chamda Raisin_Dataset.xlsx.
# PEGUEI A PLANINHA E TRANSFORMEI ELA EM UM ARQUIVO .CSV E COLOQUEI NA PASTA TRABALHO02-IA.
# IMPORTANTE,AS ULTIMAS 4 LINHAS SÃO COMENTARIOS E AO DESCOMENTAR IRA GERAR ARQUIVOS .CSV.
# BIBLIOTECAS USADAS: PANDAS E SKLEARN
#PARA BAIXAR AS BIBLIOTECAS USE: pip install pandas e pip install sklearn


#TRABALHO FEITO POR MARCELO BENTO CÔGO E ARTHUR SANTOS ALMEIDA

import pandas as pd
from sklearn.model_selection import train_test_split


#PEGANDO OS DADOS DO DATASET E ARMAZENANDO NA VARIAVEL DF.
df = pd.read_csv("Raisin_Dataset.csv")

# A FUNCAO HEAD IRA MOSTRAR AS 4 PRIMEIRAS LINHAS DE DF.
# print(df.head())

# MONTANDO DATAFRAME DOS ATRIBUTOS BÁSICOS.
X = df[['Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','Extent','Perimeter']]
Y = df[['Class']]

#AGORA FAREMOS UMA DIVISAO DE DADOS ASSIM: 30% Teste, 49% Treinamento E 21% VALIDAÇÃO

#X_resto e Y_resto ficará com os 70% restante ja que o teste pegou 30%, O PARÂMEETRO RANDOM_STATE APENAS GARANTE QUE TEREMOS SEMPRE OS MESMO NUMEROS ALEATORIOS.
#PARÂMETRO strartify=Y garante que terá ambas as classes em cada amostra
X_test, X_resto, Y_test, Y_resto = train_test_split(X,Y, test_size=0.7,random_state=38,stratify=Y) 



# Agora resta dividir a validacao e o treinamento.Pegando os 70% e dividindo em 49% de treinamento e 21% de validação.
# Para fazer isso temos que lembrar que colocar o  test_size em 0.7,pois queremos 70% de 70% para o treino.
#Dessa forma,consiguimos que o treino fique com 49% de 100% e que a validação fique com 21% do total.
X_valid, X_train, Y_valid, Y_train = train_test_split(X_resto,Y_resto, test_size=0.7,random_state=38,stratify=Y_resto) 



#ABAIXO ESTOU APENAS PEGANDO OS DADOS QUE FORAM ARMAZENADOS E TRANSFORMANDO EM UM ARQUIVO CSV.

# X_train.to_csv("X_train.csv",index=False,sep=";")
# X_test.to_csv("X_test.csv",index=False,sep=";")
# X_valid.to_csv("X_valid.csv",index=False,sep=";")

# Y_train.to_csv("Y_train.csv",index=False,sep=";")
# Y_test.to_csv("Y_test.csv",index=False,sep=";")
# Y_valid.to_csv("Y_valid.csv",index=False,sep=";")



