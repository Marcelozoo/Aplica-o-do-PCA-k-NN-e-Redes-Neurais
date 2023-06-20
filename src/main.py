from dataframe import DataFrame
from pca import Pca
from k_nn import Knn

def main() -> None:
    
    dados_frame = DataFrame()
    dados_frame.ler_csv()
    dados_frame.montar_dataframe()
    dados_frame.dividir_dados_em_treino_teste_validacao()

    pca = Pca(dados_frame)
    pca.normalizar_peso_dados()
    pca.descobrindo_variancia_dos_dados()
    pca.definindo_numero_de_componentes_com_maior_variancia()
    pca.aplica_pca_no_conjunto_x_train()

    knn = Knn(dados_frame, pca)
    knn.descobrir_melhor_combinacao_parametros()
    knn.imprimir_melhores_parametros()
    knn.cria_modelo_com_os_melhores_parametros()
    knn.treinar_modelo()
    knn.realiza_previsoes()
    knn.calcular_precisao_modelo()

if __name__=="__main__":
    main()