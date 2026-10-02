# Apollo Solutions - Machine Learning Developer Test

Solução desenvolvida para o teste técnico de Machine Learning Developer da Apollo Solutions.

O projeto utiliza embeddings de imagens para classificação de síndromes genéticas. Os embeddings possuem 320 dimensões e são classificados utilizando KNN, comparando as distâncias Euclidiana e Cosseno.

## Sobre o projeto

O dataset utilizado possui uma estrutura hierárquica organizada por síndrome, indivíduo e imagem:

```text
syndrome_id
    └── subject_id
        └── image_id
            └── embedding

Para realizar as análises e os experimentos, essa estrutura foi transformada em um DataFrame, mantendo as informações de síndrome, indivíduo, imagem e embedding.

O dataset possui:

10 síndromes
1.116 imagens
941 indivíduos
embeddings com 320 dimensões

Durante a preparação dos dados também foram verificadas dimensões dos embeddings, valores NaN/inf, imagens duplicadas, embeddings duplicados e indivíduos associados a mais de uma síndrome.

Estrutura do projeto
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
src/data.py

Responsável pelo carregamento, transformação e validação do dataset.

As principais etapas são:

carregar o arquivo pickle;
transformar o dicionário em DataFrame;
validar a integridade dos dados;
preparar os embeddings, rótulos e grupos para os experimentos.
src/knn.py

Contém a implementação da classificação utilizando KNeighborsClassifier do scikit-learn.

Foram testadas duas métricas de distância:

Euclidiana;
Cosseno.

Foram avaliados valores de k entre 1 e 15.

src/metricas.py

Contém as métricas implementadas manualmente para o projeto:

F1-Macro;
AUC ROC;
Top-1 Accuracy;
Top-3 Accuracy;
Top-5 Accuracy.

A AUC foi calculada utilizando curvas ROC One-vs-Rest para as diferentes síndromes.

src/visualizacao.py

Contém as funções utilizadas para gerar os gráficos da análise exploratória, distribuição dos dados, t-SNE e curvas ROC.

Análise exploratória

A distribuição das classes foi analisada antes dos experimentos. Existe um desbalanceamento entre as síndromes: a classe com mais imagens possui 210 imagens, enquanto a menor possui 64.

Não foram utilizados oversampling, undersampling ou SMOTE. A distribuição original do dataset foi mantida.

Também foi analisada a relação entre indivíduos e imagens. Alguns indivíduos possuem mais de uma imagem, por isso o subject_id foi utilizado como grupo durante a validação cruzada.

t-SNE

Foi utilizado t-SNE para reduzir os embeddings de 320 dimensões para duas dimensões e visualizar a distribuição das amostras de acordo com cada síndrome.

O t-SNE foi utilizado apenas para análise visual. A avaliação da classificação foi realizada utilizando o KNN.

Classificação

A classificação foi realizada utilizando KNN com:

k de 1 até 15;
distância Euclidiana;
distância Cosseno;
validação cruzada em 10 folds;
StratifiedGroupKFold;
subject_id como grupo.

A utilização do subject_id como grupo evita que imagens do mesmo indivíduo apareçam simultaneamente nos conjuntos de treinamento e teste.

O F1-Macro foi utilizado como critério para selecionar o melhor valor de k.

Resultados

Os melhores resultados encontrados foram:

Distância	k	F1-Macro	AUC ROC	Top-1	Top-3	Top-5
Euclidiana	13	0,7078 ± 0,0688	0,9390 ± 0,0148	73,21%	90,51%	95,25%
Cosseno	14	0,7488 ± 0,0414	0,9536 ± 0,0153	77,69%	92,48%	96,24%

A distância Cosseno apresentou resultados melhores que a Euclidiana nas configurações selecionadas. Com k=14, chegou a 0,7488 de F1-Macro e 0,9536 de AUC ROC.

Como executar
1. Clonar o repositório
git clone <URL_DO_REPOSITORIO>
cd <NOME_DO_REPOSITORIO>
2. Criar um ambiente virtual
python -m venv .venv

No Windows:

.venv\Scripts\activate

No Linux/macOS:

source .venv/bin/activate
3. Instalar as dependências
pip install -r requirements.txt
4. Adicionar o dataset

Coloque o arquivo:

mini_gm_public_v0.1.p

dentro da pasta:

dataset/

A estrutura esperada é:

dataset/
└── mini_gm_public_v0.1.p
5. Executar o projeto

Na raiz do projeto:

python main.py

O script executa a preparação dos dados, análise exploratória, geração do t-SNE, classificação com KNN, validação cruzada e cálculo das métricas.

Os gráficos são salvos em:

outputs/figuras/

E os resultados numéricos em:

outputs/resultados/
Principais arquivos gerados
Figuras
outputs/figuras/
├── imagens_por_sindrome.png
├── pacientes_por_sindrome.png
├── imagens_por_paciente.png
├── imagens_por_paciente_por_sindrome.png
├── distribuicao_classes.png
├── pacientes_vs_imagens.png
├── imagens_por_paciente_boxplot.png
├── distribuicao_embeddings.png
├── tsne.png
└── comparacao_roc.png
Resultados
outputs/resultados/
├── estatisticas_embeddings.csv
├── coordenadas_tsne.csv
├── resultados_knn.csv
├── resultados_knn.md
├── melhores_resultados.csv
└── melhores_resultados.md
Próximos passos

Como possíveis melhorias, seria interessante testar outros modelos de classificação, como Random Forest, SVM e XGBoost, utilizando os mesmos embeddings. Também seria possível analisar as métricas individualmente para cada síndrome e testar outras técnicas de redução de dimensionalidade, como PCA ou UMAP.

Outra possibilidade seria automatizar ainda mais os experimentos para facilitar a comparação entre diferentes modelos, métricas e configurações.

Observação

Este projeto foi desenvolvido como solução para um teste técnico de Machine Learning Developer, com foco em análise de embeddings, classificação e avaliação de modelos.