import pickle
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

def carregar_dataset(caminho):
    # Carrega o dataset em formato pickle

    #data = pd.read()
    
    with open(caminho, "rb") as arquivo:
        dados = pickle.load(arquivo)

    return dados



def achatar_dataset(dados):
    # Converte o dataset em formato de dicionário para uma estrutura tabular

    # Uma síndrome pode ter vários pacientes
    linhas = [
        (id_sindrome, id_paciente, id_imagem, np.asarray(embedding, dtype=np.float64))
        for id_sindrome, pacientes in dados.items()
        for id_paciente, imagens in pacientes.items()
        for id_imagem, embedding in imagens.items()
    ]

    return pd.DataFrame(linhas, columns=["syndrome_id", "subject_id", "image_id", "embedding"])



def validar_dataset(df, dimensao_embedding=320):
    # Aqui eu verifico se os dados estão íntegros, analisar se precisa pré-processar os dados

    tamanhos_embeddings = df["embedding"].apply(len)

    dimensoes_invalidas = (tamanhos_embeddings != dimensao_embedding).sum()

    valores_invalidos = (~df["embedding"].apply(lambda vetor: np.isfinite(vetor).all())).sum()

    imagens_ids_duplicadas = df["image_id"].duplicated().sum()

    embeddings_duplicados = pd.DataFrame(np.stack(df["embedding"].to_numpy())).duplicated().sum()

    pacientes_multiplas_sindromes = (df.groupby("subject_id")["syndrome_id"].nunique().gt(1).sum())

    print("### Validação do dataset ###")

    print(f"Embeddings com dimensão diferente de {dimensao_embedding}: {dimensoes_invalidas}")
    print(f"Embeddings com NaN/inf: {valores_invalidos}")
    print(f"image_id duplicados: {imagens_ids_duplicadas}")
    print(f"Embeddings idênticos: {embeddings_duplicados}")
    print(f"Subjects em mais de uma síndrome: {pacientes_multiplas_sindromes}")
    print(f"\nShape do DataFrame: {df.shape}")
    print(f"Número de síndromes: {df['syndrome_id'].nunique()}")
    print(f"Número de subjects: {df['subject_id'].nunique()}")


    return {
        "dimensoes_invalidas": dimensoes_invalidas,
        "valores_invalidos": valores_invalidos,
        "imagens_ids_duplicadas": imagens_ids_duplicadas,
        "embeddings_duplicados": embeddings_duplicados,
        "pacientes_multiplas_sindromes": pacientes_multiplas_sindromes,
    }
    


def preparar_dados(df):
    # Preparando embeddings, rotulos, grupos e classes

    embeddings = np.stack(df["embedding"].to_numpy())
    rotulos = df["syndrome_id"].to_numpy()

    grupos = df["subject_id"].to_numpy()

    classes = np.unique(rotulos)

    classe_para_indice = {classe: indice for indice, classe in enumerate(classes)}

    rotulos_codificados = np.array([classe_para_indice[rotulo] for rotulo in rotulos])
    
    

    return embeddings, rotulos, rotulos_codificados, grupos, classes