import pandas as pd
from sklearn.model_selection import train_test_split

class DataFrame:

    def ler_csv(self) -> None:
        """
        Essa função pega os dados do raisin_dataset e armazena em uma variável.
        """
        self.__df = pd.read_csv("dataset/Raisin_Dataset.csv")

    def montar_dataframe(self)-> None:
        """
        Essa função monta o dataframe X e Y com os atributos do raisin_dataset.
        """
        self.__X = self.__df[['Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','Extent','Perimeter']]
        self.__Y = self.__df[['Class']]

    def dividir_dados_em_treino_teste_validacao(self) -> None:

        """
        Essa função faz a divisão dos dados em : 30% Teste, 49% Treinamento e 21% validação, garantindo
        que cada amostra tenha todas as classes.
        """
        self.__X_test, X_resto, self.__Y_test, Y_resto = train_test_split(self.__X, self.__Y, test_size=0.7, random_state=38, stratify=self.__Y) 
        self.__X_valid, self.__X_train, self.__Y_valid, self.__Y_train = train_test_split(X_resto, Y_resto, test_size=0.7,random_state=38, stratify=Y_resto) 

    def get_dataframe_X(self) -> pd:
        return self.__X

    def get_x_train(self) -> pd:
        return self.__X_train
    
    def get_x_valid(self) -> pd:
        return self.__X_valid

    def get_x_test(self) -> pd:
        return self.__X_test
    
    def get_y_test(self) -> pd:
        return self.__Y_test
    
    def get_y_valid(self) -> pd:
        return self.__Y_valid
    
    def get_y_train(self) -> pd:
        return self.__Y_train
    
    