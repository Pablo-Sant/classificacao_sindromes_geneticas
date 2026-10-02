import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE


def plotar_imagens_por_sindrome(df):
    quantidades = df.groupby("syndrome_id").size().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))

    sns.barplot(x=quantidades.index.astype(str), y=quantidades.values)

    plt.title("Número de imagens por síndrome")
    plt.xlabel("Síndrome")
    plt.ylabel("Número de imagens")

    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt.gcf()


def plotar_pacientes_por_sindrome(df):
    quantidades = df.groupby("syndrome_id")["subject_id"].nunique().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))

    sns.barplot(x=quantidades.index.astype(str), y=quantidades.values)

    plt.title("Número de indivíduos por síndrome")
    plt.xlabel("Síndrome")
    plt.ylabel("Número de indivíduos")

    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt.gcf()


def plotar_imagens_por_paciente(df):
    imagens_por_paciente = df.groupby("subject_id").size()

    plt.figure(figsize=(8, 5))

    sns.histplot(imagens_por_paciente, discrete=True)

    plt.title("Distribuição do número de imagens por indivíduo")
    plt.xlabel("Número de imagens")
    plt.ylabel("Número de indivíduos")

    plt.tight_layout()

    return plt.gcf()


def plotar_imagens_por_paciente_por_sindrome(df):
    dados = df.groupby(["syndrome_id", "subject_id"]).size().reset_index(name="numero_imagens")

    plt.figure(figsize=(12, 6))

    sns.boxplot(data=dados, x="syndrome_id", y="numero_imagens")

    plt.title("Número de imagens por indivíduo em cada síndrome")
    plt.xlabel("Síndrome")
    plt.ylabel("Número de imagens por indivíduo")

    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt.gcf()


def plotar_distribuicao_classes(df):
    distribuicao = (
        df.groupby("syndrome_id")
        .agg(
            imagens=("image_id", "count"),
            pacientes=("subject_id", "nunique")
        )
        .sort_values("imagens", ascending=False)
    )

    distribuicao.plot(kind="bar", figsize=(11, 6))

    plt.title("Distribuição de imagens e indivíduos por síndrome")
    plt.xlabel("Síndrome")
    plt.ylabel("Quantidade")

    plt.xticks(rotation=45)
    plt.legend(title="")

    plt.tight_layout()

    return plt.gcf(), distribuicao


def plotar_distribuicao_embeddings(embeddings):
    valores_embeddings = embeddings.flatten()

    plt.figure(figsize=(10, 5))

    sns.violinplot(y=valores_embeddings)

    plt.title("Distribuição dos valores dos embeddings")
    plt.ylabel("Valor")

    plt.tight_layout()

    return plt.gcf()


def plotar_relacao_pacientes_imagens(df):
    distribuicao = (
        df.groupby("syndrome_id")
        .agg(
            imagens=("image_id", "count"),
            pacientes=("subject_id", "nunique")
        )
    )

    plt.figure(figsize=(8, 6))

    sns.scatterplot(data=distribuicao, x="pacientes", y="imagens", s=100)

    for sindrome, linha in distribuicao.iterrows():

        plt.annotate(
            str(sindrome),
            (linha["pacientes"], linha["imagens"]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.title("Relação entre indivíduos e imagens por síndrome")
    plt.xlabel("Número de indivíduos")
    plt.ylabel("Número de imagens")

    plt.tight_layout()

    return plt.gcf()


def plotar_boxplot_imagens_por_paciente(df):
    imagens_por_paciente = df.groupby("subject_id").size()

    plt.figure(figsize=(8, 5))

    sns.boxplot(y=imagens_por_paciente)

    plt.title("Distribuição do número de imagens por indivíduo")
    plt.ylabel("Número de imagens")

    plt.tight_layout()

    return plt.gcf()


def executar_tsne(embeddings, df, estado_aleatorio):
    """
    Executa t-SNE e retorna o DataFrame
    com as coordenadas.
    """

    tsne = TSNE(n_components=2, random_state=estado_aleatorio)

    coordenadas_tsne = tsne.fit_transform(embeddings)

    dados_tsne = pd.DataFrame({
        "tsne_1": coordenadas_tsne[:, 0],
        "tsne_2": coordenadas_tsne[:, 1],
        "syndrome_id": df["syndrome_id"].astype(str).values,
        "subject_id": df["subject_id"].values
    })

    return dados_tsne


def plotar_tsne(dados_tsne):
    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=dados_tsne,
        x="tsne_1",
        y="tsne_2",
        hue="syndrome_id",
        palette="tab10",
        s=45,
        alpha=0.8
    )

    plt.title("t-SNE dos embeddings por síndrome")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.legend(title="Síndrome", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    return plt.gcf()


def plotar_comparacao_roc(fpr_euclidiana, tpr_euclidiana, auc_euclidiana, fpr_cosseno, tpr_cosseno, auc_cosseno):
    plt.figure(figsize=(10, 7))

    plt.plot(
        fpr_euclidiana,
        tpr_euclidiana,
        label=f"Euclidiana (AUC = {auc_euclidiana:.3f})"
    )

    plt.plot(
        fpr_cosseno,
        tpr_cosseno,
        label=f"Cosseno (AUC = {auc_cosseno:.3f})"
    )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Aleatório")

    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")

    plt.title("Curva ROC média - KNN")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt.gcf()