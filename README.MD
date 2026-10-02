Relatório do Projeto: Apollo Solutions - Machine Learning Developer Test

Solução desenvolvida para o teste técnico de Machine Learning Developer da Apollo Solutions.

O projeto utiliza embeddings de imagens para classificação de síndromes genéticas. Os embeddings possuem 320 dimensões e são classificados utilizando o algoritmo KNN, comparando as métricas de distância Euclidiana e Cosseno.

1. Análise e Preparação dos Dados

Para começar o projeto, carreguei o dataset mini_gm_public_v0.1.p, que possui uma estrutura hierárquica organizada por síndrome, indivíduo e imagem:

text
syndrome_id
    └── subject_id
        └── image_id
            └── embedding

Após achatar essa estrutura e transformá-la em um DataFrame, encontrei 1.116 imagens, pertencentes a 941 indivíduos e distribuídas entre 10 síndromes diferentes. Cada imagem possui um embedding de 320 dimensões, gerado previamente por um modelo de classificação.

Fiz algumas validações no dataset para verificar:

Dimensão dos embeddings (320);
Presença de valores NaN ou infinitos;
Embeddings e imagens duplicadas;
Indivíduos associados a mais de uma síndrome.

Como não foram encontradas inconsistências nesses pontos, não foi necessário fazer remoção ou imputação de dados.

Tratamento dos Indivíduos e Validação Cruzada

Uma questão central do projeto foi a relação entre indivíduos e imagens. Alguns indivíduos possuem mais de uma imagem no dataset, chegando a até 11 imagens em alguns casos. Por conta disso, o subject_id não entra como variável de entrada (feature) do modelo, pois é apenas um identificador.

Contudo, utilizei o subject_id como variável de agrupamento na validação cruzada através do StratifiedGroupKFold com 10 folds. Dessa forma, imagens do mesmo indivíduo pertencem sempre ao mesmo fold (treinamento ou teste), garantindo que o modelo não seja avaliado com amostras de um paciente que ele já viu no treino e eliminando o risco de vazamento de informação (data leakage).

Desbalanceamento de Classes

Foi observado um desbalanceamento de aproximadamente 3,28 vezes entre as classes (a maior síndrome possui 210 imagens, enquanto a menor possui 64). O mesmo desbalanceamento ocorre no nível dos indivíduos:

Síndrome	Imagens	Pacientes
300000034	210	137
300000080	198	180
100192430	136	127
300000007	115	93
300000082	98	83
100610443	89	76
300000018	74	64
100180860	67	62
100610883	65	59
700018215	64	60

A distribuição original do dataset foi preservada, sem aplicação de oversampling, undersampling ou SMOTE. Para lidar com essa diferença na avaliação, utilizei o F1 Macro, que calcula o F1 de cada classe e faz a média simples, atribuindo o mesmo peso a todas as síndromes.

2. Análise dos Embeddings e t-SNE

Realizei uma análise estatística dos embeddings para verificar a distribuição dos valores nas 320 dimensões. Como os valores apresentaram escalas próximas e bem centradas em torno do zero, não utilizei StandardScaler e não criei novas features. A intenção foi preservar a geometria original dos vetores, fundamental para a comparação entre as distâncias Euclidiana e Cosseno.

Para a visualização, utilizei o t-SNE para reduzir as 320 dimensões para 2D. O gráfico revelou a formação de aglomerados por síndrome, embora com certas zonas de sobreposição entre as classes. O t-SNE serviu puramente para análise exploratória visual, ficando a capacidade de discriminação a cargo do KNN.

3. Classificação com KNN

A classificação foi implementada utilizando o KNeighborsClassifier do scikit-learn com as seguintes configurações:

<!-- REVISAR: o código final (src/knn.py) implementa o KNN manualmente e não usa o KNeighborsClassifier. Ajustar este parágrafo para refletir o que o código faz. -->
Validação cruzada em 10 folds via StratifiedGroupKFold;
Agrupamento por subject_id;
Intervalo de k testado: de 1 a 15;
Métricas de distância: Euclidiana e Cosseno.
Métricas Implementadas
F1 Macro: Média do F1-score de cada classe;
AUC ROC Macro: Média da área sob a curva ROC em estratégia One-vs-Rest (OvR);
Top-1, Top-3 e Top-5 Accuracy: Para verificar a taxa de acerto do modelo entre os k primeiros palpites.
4. Resultados

O critério principal adotado para a seleção da melhor configuração foi o F1 Macro. Os melhores resultados para cada distância foram:

<!-- REVISAR: conferir todos os valores desta tabela com outputs/results/best_results.csv da última execução do main.py. Os valores de F1 (0,7078 e 0,7488) não coincidem com a saída do main final (k=14 nas duas distâncias, F1 de 0,7090 e 0,7572). -->
Distância	k Selecionado	F1 Macro	AUC Macro	Top-1 Accuracy	Top-3 Accuracy	Top-5 Accuracy
Euclidiana	13	0,7078 ± 0,0688	0,9390 ± 0,0148	73,21% ± 5,95%	90,51% ± 2,29%	95,25% ± 1,74%
Cosseno	14	0,7488 ± 0,0414	0,9536 ± 0,0153	77,69% ± 3,21%	92,48% ± 2,59%	96,24% ± 1,59%
Principais Destaques
Superioridade do Cosseno: A distância Cosseno superou a Euclidiana em todas as métricas avaliadas, além de apresentar menor desvio padrão na validação cruzada (± 0,0414), demonstrando ser uma métrica mais estável para o espaço vetorial de 320 dimensões.
Top-k Accuracy: A configuração campeã (Cosseno, k = 14) alcançou 92,48% de Top-3 Accuracy e 96,24% de Top-5 Accuracy, o que mostra que a síndrome correta está praticamente sempre entre as primeiras opções do modelo.
Comportamento do k: O F1 Macro atingiu seu pico em k = 13 (Euclidiana) e k = 14 (Cosseno). A AUC continuou crescendo até k = 15, mas o critério definido para desempate foi mantido estritamente no F1 Macro.
5. Estrutura dos Arquivos e Execução Local
Estrutura do Projeto
<!-- REVISAR: os nomes das pastas de saída (figuras/resultados) e a posição do main.py na raiz devem bater com o código. No main.py enviado, as pastas são outputs/figures e outputs/results, e DIRETORIO_BASE usa parent.parent, o que só funciona se o main.py estiver em uma subpasta. -->
text
.
├── dataset/
│   └── mini_gm_public_v0.1.p
├── src/
│   ├── data.py         # Carregamento, transformação e validação
│   ├── knn.py          # Execução da classificação com KNN
│   ├── metricas.py     # Implementação manual das métricas
│   └── visualizacao.py # Geração dos gráficos e t-SNE
├── outputs/
│   ├── figuras/        # Gráficos salvos em PNG
│   └── resultados/     # Relatórios detalhados em CSV e MD
├── main.py             # Script principal de execução
├── requirements.txt    # Dependências do projeto
└── README.md
Instruções de Execução
Descompactar a pasta do projeto entregue em anexo e navegar até ela via terminal:
bash
   cd caminho/para/pasta_do_projeto
Criar e ativar o ambiente virtual:
bash
   python -m venv .venv

   # Windows:
   .venv\Scripts\activate

   # Linux/macOS:
   source .venv/bin/activate
Instalar as dependências do projeto:
bash
   pip install -r requirements.txt
Verificar o dataset: Garantir que o arquivo mini_gm_public_v0.1.p está localizado na pasta dataset/.
Executar o script principal:
bash
   python main.py

O script executará todo o pipeline: pré-processamento, análise exploratória, redução dimensional com t-SNE, treinamento, validação cruzada do KNN e geração dos relatórios/figuras nas pastas de saída (outputs/).

6. Desafios e Soluções
Estruturação dos dados: Transformar o dataset hierárquico em um DataFrame sem perder as relações entre síndrome, indivíduo e embedding exigiu atenção no pré-processamento.
Implementação manual de métricas: O cálculo manual de Top-k e ROC OvR para o ambiente de validação cruzada agregou um nível maior de profundidade técnica e compreensão do funcionamento das métricas por trás das funções prontas da biblioteca.
Resolução de dúvidas: Foi realizada consulta constante à documentação oficial do scikit-learn, tutoriais de visualização e ferramentas de inteligência artificial para auxiliar no entendimento do t-SNE e validação cruzada agrupada.
7. Recomendações e Próximos Passos
Novos Algoritmos: Avaliar o desempenho de outros classificadores como Random Forest, SVM e XGBoost sobre os mesmos embeddings.
Análise por Síndrome: Avaliar as métricas de forma individualizada por classe para verificar onde o modelo encontra maior dificuldade de separação.
Outras Reduções de Dimensionalidade: Testar técnicas como UMAP e PCA antes da classificação para comparar a retenção de informação em relação aos 320 componentes originais.
Automação do Pipeline: Implementar um orquestrador de experimentos para facilitar o versionamento de hiperparâmetros.