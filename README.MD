# Apollo Solutions - Machine Learning Developer Test

Solução desenvolvida para o teste técnico de **Machine Learning Developer da Apollo Solutions**.

O projeto utiliza **embeddings de imagens com 320 dimensões** para classificação de **10 síndromes genéticas** utilizando **KNN**, comparando as distâncias **Euclidiana** e **Cosseno**.

---

## Dataset

O dataset possui uma estrutura hierárquica:

```text
syndrome_id
└── subject_id
    └── image_id
        └── embedding
```

Após a transformação para `DataFrame`, foram obtidos:

* **10 síndromes**
* **1.116 imagens**
* **941 indivíduos**
* **320 dimensões por embedding**

Foram realizadas validações de dimensionalidade, valores `NaN`/`inf`, duplicidades e indivíduos associados a múltiplas síndromes.

Como alguns indivíduos possuem várias imagens, o `subject_id` foi utilizado como **grupo na validação cruzada**, evitando que imagens do mesmo indivíduo aparecessem simultaneamente em treinamento e teste.

O dataset apresenta desbalanceamento entre as classes. A distribuição original foi mantida, sem utilização de oversampling, undersampling ou SMOTE.

---

## Metodologia

### Classificação

Foi utilizado `KNeighborsClassifier`, testando:

* distância Euclidiana;
* distância Cosseno;
* `k` de 1 a 15;
* **10-fold StratifiedGroupKFold**;
* `subject_id` como grupo.

O **F1-Macro** foi utilizado como critério para seleção do melhor `k`.

### Métricas

Foram avaliadas:

* F1-Macro;
* AUC ROC;
* Top-1 Accuracy;
* Top-3 Accuracy;
* Top-5 Accuracy.

As métricas foram implementadas manualmente, incluindo a AUC através de curvas ROC **One-vs-Rest**.

### Análise dos embeddings

Foi utilizado **t-SNE** para reduzir os embeddings de 320 para 2 dimensões e visualizar a distribuição das síndromes.

O t-SNE foi utilizado somente para visualização; a classificação foi realizada sobre os embeddings originais.

---

## Resultados

| Distância  |  k |        F1-Macro |         AUC ROC |  Top-1 |  Top-3 |  Top-5 |
| ---------- | -: | --------------: | --------------: | -----: | -----: | -----: |
| Euclidiana | 13 | 0,7078 ± 0,0688 | 0,9390 ± 0,0148 | 73,21% | 90,51% | 95,25% |
| Cosseno    | 14 | 0,7488 ± 0,0414 | 0,9536 ± 0,0153 | 77,69% | 92,48% | 96,24% |

A configuração selecionada pela métrica **F1-Macro** para a distância Cosseno foi `k=14`.

> A maior AUC não ocorreu necessariamente no mesmo `k` que o maior F1. A seleção foi mantida baseada no F1-Macro, conforme definido na metodologia.

---

## Estrutura do projeto

```text
.
├── dataset/
│   └── mini_gm_public_v0.1.p
│
├── src/
│   ├── data.py
│   ├── knn.py
│   ├── metricas.py
│   └── visualizacao.py
│
├── outputs/
│   ├── figuras/
│   └── resultados/
│
├── main.py
├── requirements.txt
└── README.md
```

### Principais módulos

* `data.py` — carregamento, transformação e validação dos dados.
* `knn.py` — experimentos de classificação com KNN.
* `metricas.py` — implementação das métricas.
* `visualizacao.py` — geração das visualizações.
* `main.py` — execução do pipeline completo.

---

## Como executar

### 1. Criar o ambiente virtual

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Adicionar o dataset

Coloque `mini_gm_public_v0.1.p` em:

```text
dataset/
└── mini_gm_public_v0.1.p
```

### 4. Executar

```bash
python main.py
```

Os gráficos serão gerados em:

```text
outputs/figuras/
```

E os resultados em:

```text
outputs/resultados/
```

---

## Possíveis melhorias

* Comparar KNN com **Random Forest, SVM e XGBoost**.
* Analisar métricas individualmente por síndrome.
* Testar **PCA** e **UMAP**.
* Explorar outros valores de `k` e configurações do KNN.
* Automatizar a comparação entre diferentes modelos e hiperparâmetros.
