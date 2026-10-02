from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.data import carregar_dataset, achatar_dataset, validar_dataset, preparar_dados
from src.knn import prever_knn
from src.metricas import f1_macro, acuracia_top_k, auc_multiclasse, roc_multiclasse
from src.visualizacao import (
    plotar_imagens_por_sindrome,
    plotar_pacientes_por_sindrome,
    plotar_imagens_por_paciente,
    plotar_imagens_por_paciente_por_sindrome,
    plotar_distribuicao_classes,
    plotar_distribuicao_embeddings,
    plotar_relacao_pacientes_imagens,
    plotar_boxplot_imagens_por_paciente,
    executar_tsne,
    plotar_tsne,
    plotar_comparacao_roc
)

# Configuração de diretórios
DIRETORIO_BASE = Path(__file__).resolve().parent.parent
CAMINHO_DATASET = DIRETORIO_BASE / "dataset" / "mini_gm_public_v0.1.p"
DIRETORIO_SAIDA = DIRETORIO_BASE / "outputs"
DIRETORIO_FIGURAS = DIRETORIO_SAIDA / "figuras"
DIRETORIO_RESULTADOS = DIRETORIO_SAIDA / "resultados"

DIRETORIO_FIGURAS.mkdir(parents=True, exist_ok=True)
DIRETORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)


# Carregamento, preparação e validação do dataset
dados = carregar_dataset(CAMINHO_DATASET)
df = achatar_dataset(dados)
validar_dataset(df)


# EDA e Visualizações
figura = plotar_imagens_por_sindrome(df)
figura.savefig(DIRETORIO_FIGURAS / "imagens_por_sindrome.png", dpi=300, bbox_inches="tight")

figura = plotar_pacientes_por_sindrome(df)
figura.savefig(DIRETORIO_FIGURAS / "pacientes_por_sindrome.png", dpi=300, bbox_inches="tight")

figura = plotar_imagens_por_paciente(df)
figura.savefig(DIRETORIO_FIGURAS / "imagens_por_paciente.png", dpi=300, bbox_inches="tight")

figura = plotar_imagens_por_paciente_por_sindrome(df)
figura.savefig(DIRETORIO_FIGURAS / "imagens_por_paciente_por_sindrome.png", dpi=300, bbox_inches="tight")

figura, distribuicao_classes = plotar_distribuicao_classes(df)
figura.savefig(DIRETORIO_FIGURAS / "distribuicao_classes.png", dpi=300, bbox_inches="tight")

figura = plotar_relacao_pacientes_imagens(df)
figura.savefig(DIRETORIO_FIGURAS / "pacientes_vs_imagens.png", dpi=300, bbox_inches="tight")

figura = plotar_boxplot_imagens_por_paciente(df)
figura.savefig(DIRETORIO_FIGURAS / "imagens_por_paciente_boxplot.png", dpi=300, bbox_inches="tight")


# Preparação dos embeddings
embeddings, rotulos, rotulos_codificados, grupos, classes = preparar_dados(df)

# Estatísticas dos embeddings
estatisticas_embeddings = pd.DataFrame(embeddings).describe().T
estatisticas_embeddings[["mean", "std", "min", "max"]].to_csv(
    DIRETORIO_RESULTADOS / "estatisticas_embeddings.csv"
)

figura = plotar_distribuicao_embeddings(embeddings)
figura.savefig(DIRETORIO_FIGURAS / "distribuicao_embeddings.png", dpi=300, bbox_inches="tight")


# t-SNE
dados_tsne = executar_tsne(embeddings, df, 42)
dados_tsne.to_csv(DIRETORIO_RESULTADOS / "coordenadas_tsne.csv", index=False)

figura = plotar_tsne(dados_tsne)
figura.savefig(DIRETORIO_FIGURAS / "tsne.png", dpi=300, bbox_inches="tight")


# KNN e Validação Cruzada
resultados_knn = []
numero_classes = len(classes)

for metrica in ["euclidiana", "cosseno"]:
    for k in range(1, 16):

        f1_por_fold, auc_por_fold = [], []
        top1_por_fold, top3_por_fold, top5_por_fold = [], [], []

        validacao = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

        for indices_treino, indices_teste in validacao.split(embeddings, rotulos_codificados, grupos):
            dados_treino = embeddings[indices_treino]
            dados_teste = embeddings[indices_teste]

            rotulos_treino = rotulos_codificados[indices_treino]
            rotulos_teste = rotulos_codificados[indices_teste]

            previsoes, pontuacoes, classificacoes = prever_knn(
                dados_treino, rotulos_treino, dados_teste, k, metrica
            )

            f1_por_fold.append(f1_macro(rotulos_teste, previsoes, numero_classes))
            auc_por_fold.append(auc_multiclasse(rotulos_teste, pontuacoes, numero_classes))
            top1_por_fold.append(acuracia_top_k(rotulos_teste, classificacoes, k=1))
            top3_por_fold.append(acuracia_top_k(rotulos_teste, classificacoes, k=3))
            top5_por_fold.append(acuracia_top_k(rotulos_teste, classificacoes, k=5))

        resultados_knn.append({
            "metrica": metrica,
            "k": k,
            "f1_media": np.mean(f1_por_fold),
            "f1_desvio": np.std(f1_por_fold),
            "auc_media": np.mean(auc_por_fold),
            "auc_desvio": np.std(auc_por_fold),
            "top1_media": np.mean(top1_por_fold),
            "top1_desvio": np.std(top1_por_fold),
            "top3_media": np.mean(top3_por_fold),
            "top3_desvio": np.std(top3_por_fold),
            "top5_media": np.mean(top5_por_fold),
            "top5_desvio": np.std(top5_por_fold)
        })


# --- PROCESSAMENTO E EXPORTAÇÃO DOS RESULTADOS GERAIS DO KNN ---
resultados_df = pd.DataFrame(resultados_knn)

# 1. Guarda o CSV original completo (alta precisão para consultas futuras)
resultados_df.to_csv(DIRETORIO_RESULTADOS / "resultados_knn.csv", index=False)

# 2. Reestrutura para o Markdown (formato compacto Média ± Desvio com 4 casas decimais)
df_knn_fmt = resultados_df.round(4)
df_knn_exibicao = pd.DataFrame({
    "Métrica": df_knn_fmt["metrica"],
    "k": df_knn_fmt["k"],
    "F1-Macro": df_knn_fmt["f1_media"].astype(str) + " ± " + df_knn_fmt["f1_desvio"].astype(str),
    "AUC ROC": df_knn_fmt["auc_media"].astype(str) + " ± " + df_knn_fmt["auc_desvio"].astype(str),
    "Top-1 Acc": df_knn_fmt["top1_media"].astype(str) + " ± " + df_knn_fmt["top1_desvio"].astype(str),
    "Top-3 Acc": df_knn_fmt["top3_media"].astype(str) + " ± " + df_knn_fmt["top3_desvio"].astype(str),
    "Top-5 Acc": df_knn_fmt["top5_media"].astype(str) + " ± " + df_knn_fmt["top5_desvio"].astype(str)
})

with open(DIRETORIO_RESULTADOS / "resultados_knn.md", "w", encoding="utf-8") as f:
    f.write(df_knn_exibicao.to_string(index=False))


# --- FILTRO E EXPORTAÇÃO DOS MELHORES RESULTADOS POR F1 ---
melhores_resultados = []
for metrica in resultados_df["metrica"].unique():
    resultados_metrica = resultados_df[resultados_df["metrica"] == metrica]
    melhor = resultados_metrica.loc[resultados_metrica["f1_media"].idxmax()]
    melhores_resultados.append(melhor)

melhores_resultados_df = pd.DataFrame(melhores_resultados)

# 1. Guarda o CSV original dos melhores resultados
melhores_resultados_df.to_csv(DIRETORIO_RESULTADOS / "melhores_resultados.csv", index=False)

# 2. Reestrutura o Markdown dos melhores resultados para relatórios
df_melhores_fmt = melhores_resultados_df.round(4)
df_melhores_exibicao = pd.DataFrame({
    "Métrica": df_melhores_fmt["metrica"],
    "Best k": df_melhores_fmt["k"],
    "F1-Macro": df_melhores_fmt["f1_media"].astype(str) + " ± " + df_melhores_fmt["f1_desvio"].astype(str),
    "AUC ROC": df_melhores_fmt["auc_media"].astype(str) + " ± " + df_melhores_fmt["auc_desvio"].astype(str),
    "Top-1 Acc": df_melhores_fmt["top1_media"].astype(str) + " ± " + df_melhores_fmt["top1_desvio"].astype(str),
    "Top-3 Acc": df_melhores_fmt["top3_media"].astype(str) + " ± " + df_melhores_fmt["top3_desvio"].astype(str),
    "Top-5 Acc": df_melhores_fmt["top5_media"].astype(str) + " ± " + df_melhores_fmt["top5_desvio"].astype(str)
})

with open(DIRETORIO_RESULTADOS / "melhores_resultados.md", "w", encoding="utf-8") as f:
    f.write(df_melhores_exibicao.to_string(index=False))


# Mapeamento do melhor K por métrica
melhores_k = {linha["metrica"]: int(linha["k"]) for _, linha in melhores_resultados_df.iterrows()}


# Curvas ROC para os melhores modelos
resultados_roc = {}

for metrica in ["euclidiana", "cosseno"]:
    k = melhores_k[metrica]
    validacao = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

    fpr_medio = np.linspace(0, 1, 100)
    tprs_por_fold, aucs_por_fold = [], []

    for indices_treino, indices_teste in validacao.split(embeddings, rotulos_codificados, grupos):
        dados_treino = embeddings[indices_treino]
        dados_teste = embeddings[indices_teste]

        rotulos_treino = rotulos_codificados[indices_treino]
        rotulos_teste = rotulos_codificados[indices_teste]

        previsoes, pontuacoes, classificacoes = prever_knn(
            dados_treino, rotulos_treino, dados_teste, k, metrica
        )

        curvas_classes = roc_multiclasse(rotulos_teste, pontuacoes, numero_classes)

        tprs_das_classes, aucs_das_classes = [], []
        for curva in curvas_classes:
            tpr_interpolado = np.interp(
                fpr_medio,
                curva["taxas_falsos_positivos"],
                curva["taxas_verdadeiros_positivos"]
            )
            tprs_das_classes.append(tpr_interpolado)
            aucs_das_classes.append(curva["auc"])

        tprs_por_fold.append(np.mean(tprs_das_classes, axis=0))
        aucs_por_fold.append(np.mean(aucs_das_classes))

    resultados_roc[metrica] = {
        "fpr": fpr_medio,
        "tpr": np.mean(tprs_por_fold, axis=0),
        "auc": np.mean(aucs_por_fold),
        "k": k
    }


# Salvar gráfico da curva ROC comparativa
figura = plotar_comparacao_roc(
    resultados_roc["euclidiana"]["fpr"],
    resultados_roc["euclidiana"]["tpr"],
    resultados_roc["euclidiana"]["auc"],
    resultados_roc["cosseno"]["fpr"],
    resultados_roc["cosseno"]["tpr"],
    resultados_roc["cosseno"]["auc"]
)
figura.savefig(DIRETORIO_FIGURAS / "comparacao_roc.png", dpi=300, bbox_inches="tight")