import numpy as np


def f1_macro(rotulos_verdadeiros, rotulos_previstos, numero_classes):
    """Calcula F1 Macro manualmente"""

    pontuacoes_f1 = []

    for classe in range(numero_classes):

        verdadeiros_positivos = np.sum((rotulos_verdadeiros == classe) & (rotulos_previstos == classe))

        falsos_positivos = np.sum((rotulos_verdadeiros != classe) & (rotulos_previstos == classe))

        falsos_negativos = np.sum((rotulos_verdadeiros == classe) & (rotulos_previstos != classe))

        denominador_precisao = verdadeiros_positivos + falsos_positivos
        denominador_recall = verdadeiros_positivos + falsos_negativos

        if denominador_precisao == 0:
            precisao = 0.0
        else:
            precisao = verdadeiros_positivos / denominador_precisao

        if denominador_recall == 0:
            recall = 0.0
        else:
            recall = verdadeiros_positivos / denominador_recall

        if precisao + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precisao * recall / (precisao + recall)

        pontuacoes_f1.append(f1)

    return np.mean(pontuacoes_f1)


def acuracia_top_k(rotulos_verdadeiros, classificacoes, k):
    """Calcula Top-k Accuracy utilizando a classificação produzida pelo KNN"""

    corretos = np.array([
        rotulo_verdadeiro in classificacao[:k]
        for rotulo_verdadeiro, classificacao in zip(rotulos_verdadeiros, classificacoes)
    ])

    return np.mean(corretos)


def curva_roc_binaria(rotulos_verdadeiros, pontuacoes):
    """Calcula manualmente a curva ROC binária. rotulos_verdadeiros deve conter 0 e 1"""

    ordem = np.argsort(-pontuacoes, kind="stable")

    rotulos_ordenados = rotulos_verdadeiros[ordem]
    pontuacoes_ordenadas = pontuacoes[ordem]

    positivos = np.sum(rotulos_ordenados == 1)
    negativos = np.sum(rotulos_ordenados == 0)

    if positivos == 0 or negativos == 0:
        raise ValueError("As classes positiva e negativa devem estar presentes")

    taxas_verdadeiros_positivos = [0.0]
    taxas_falsos_positivos = [0.0]

    verdadeiros_positivos = 0
    falsos_positivos = 0

    pontuacao_anterior = None

    for rotulo, pontuacao in zip(rotulos_ordenados, pontuacoes_ordenadas):

        if pontuacao_anterior is not None and pontuacao != pontuacao_anterior:
            taxas_verdadeiros_positivos.append(verdadeiros_positivos / positivos)
            taxas_falsos_positivos.append(falsos_positivos / negativos)

        if rotulo == 1:
            verdadeiros_positivos += 1
        else:
            falsos_positivos += 1

        pontuacao_anterior = pontuacao

    taxas_verdadeiros_positivos.append(verdadeiros_positivos / positivos)
    taxas_falsos_positivos.append(falsos_positivos / negativos)

    return np.asarray(taxas_falsos_positivos), np.asarray(taxas_verdadeiros_positivos)


def auc_trapezoidal(taxas_falsos_positivos, taxas_verdadeiros_positivos):
    """Calcula AUC usando integração pelo método dos trapézios"""

    ordem = np.argsort(taxas_falsos_positivos)

    taxas_falsos_positivos = taxas_falsos_positivos[ordem]
    taxas_verdadeiros_positivos = taxas_verdadeiros_positivos[ordem]

    return np.sum(
        (taxas_falsos_positivos[1:] - taxas_falsos_positivos[:-1])
        * (taxas_verdadeiros_positivos[1:] + taxas_verdadeiros_positivos[:-1])
        / 2
    )


def auc_multiclasse(rotulos_verdadeiros, pontuacoes, numero_classes):
    """Calcula AUC para múltiplas classes utilizando One-vs-Rest e média macro"""

    aucs_classes = []

    for classe in range(numero_classes):

        rotulos_binarios = (rotulos_verdadeiros == classe).astype(int)

        taxas_falsos_positivos, taxas_verdadeiros_positivos = curva_roc_binaria(
            rotulos_binarios, pontuacoes[:, classe]
        )

        auc_classe = auc_trapezoidal( taxas_falsos_positivos, taxas_verdadeiros_positivos)

        aucs_classes.append(auc_classe)

    return np.mean(aucs_classes)


def roc_multiclasse(rotulos_verdadeiros, pontuacoes, numero_classes):
    """Calcula as curvas ROC One-vs-Rest para cada classe"""

    curvas = []

    for classe in range(numero_classes):

        rotulos_binarios = (rotulos_verdadeiros == classe).astype(int)

        taxas_falsos_positivos, taxas_verdadeiros_positivos = curva_roc_binaria(
            rotulos_binarios,
            pontuacoes[:, classe]
        )

        auc_classe = auc_trapezoidal(taxas_falsos_positivos, taxas_verdadeiros_positivos)

        curvas.append({
            "classe": classe,
            "taxas_falsos_positivos": taxas_falsos_positivos,
            "taxas_verdadeiros_positivos": taxas_verdadeiros_positivos,
            "auc": auc_classe
        })

    return curvas

