import numpy as np

from sklearn.neighbors import KNeighborsClassifier


def prever_knn(dados_treino, rotulos_treino, dados_teste, k, metrica="euclidiana"):
    """Executa KNN para todas as amostras de teste."""

    if metrica == "euclidiana":
        metrica_sklearn = "euclidean"

    elif metrica == "cosseno":
        metrica_sklearn = "cosine"

    else:
        raise ValueError("Métrica deve ser 'euclidiana' ou 'cosseno'.")

    modelo = KNeighborsClassifier(
        n_neighbors=k,
        metric=metrica_sklearn
    )

    modelo.fit(dados_treino, rotulos_treino)

    previsoes = modelo.predict(dados_teste)
    pontuacoes = modelo.predict_proba(dados_teste)
    classificacoes = np.argsort(-pontuacoes, axis=1)

    return previsoes, pontuacoes, classificacoes