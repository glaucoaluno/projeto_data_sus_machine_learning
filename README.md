# Análise de Mortalidade com Machine Learning

## Autores

**Alunos**:
- Elisangela Oliveira (CP301492X)
- Glauco Neto (CP3025845)
- Wellington Gomes (CP3025853)

**Disciplina**: Introdução ao Aprendizado de Máquina  
**Instituição**: IFSP  
**Data**: 2025

## 📋 Resumo do Projeto

Este projeto implementa técnicas de Machine Learning para prever a causa de morte com base em características demográficas e contextuais. Utiliza dados reais de mortalidade do Brasil fornecidos pelo DATASUS (Sistema de Informações sobre Mortalidade - SIM).

---

## 🎯 Objetivo

Desenvolver e comparar três modelos de classificação para prever a causa básica do óbito (CAUSABAS) a partir de variáveis demográficas e características da morte, contribuindo para a análise epidemiológica de padrões de mortalidade.

---

## 📊 Fonte de Dados

**Dataset**: Dados de Mortalidade 2024 - DATASUS  
**Origem**: https://opendatasus.saude.gov.br/  
**Formato**: CSV com delimitador `;`
**Tamanho**: ~494 MB (~1 milhão de registros)
**Colunas**: 88 variáveis

### Variáveis Utilizadas

#### **Features (Variáveis Independentes)**
- **SEXO**: Sexo do falecido (1 = Masculino, 2 = Feminino)
- **IDADE_CALCULADA**: Idade em anos (0-120)
- **RACACOR**: Raça/Cor (1 = Branca, 2 = Preta, 3 = Amarela, 4 = Parda, 5 = Indígena)
- **ESC2010**: Escolaridade (0 = Sem escolaridade até 5 = Superior completo)
- **LOCOCOR**: Local de ocorrência (1 = Hospital, 2 = Outro, 3 = Domicílio, 4 = Via pública, etc.)
- **CIRCOBITO**: Circunstância do óbito (1 = Acidente, 2 = Suicídio, 3 = Homicídio, 9 = Ignorado)

#### **Target (Variável Dependente)**
- **CAUSABAS**: Causa básica do óbito (Código CID-10)

---

## 🔧 Metodologia

### **1. Pré-processamento de Dados**

#### Carregamento
- Leitura do arquivo CSV com encoding `latin1`
- Tratamento de erros e validação de dimensões
- Verificação de disponibilidade de variáveis

#### Limpeza
- Remoção de valores ausentes (dropna)
- Seleção de 6 features demográficas e contextuais
- Filtro de dados incompletos

#### Simplificação da Variável Alvo
- Extração da primeira letra do código CID-10
- Redução de ~500 categorias para ~20 principais
- Filtro de categorias com menos de 1000 casos
- Resultado: 10-15 classes principais

#### Codificação de Variáveis
- **LabelEncoder** para variáveis categóricas
- Conversão de 5 features categóricas em valores numéricos
- Armazenamento de encoders para uso futuro

---

### **2. Preparação dos Dados**

#### Divisão Treino/Teste
- **Proporção**: 70% treino, 30% teste
- **Estratificação**: Mantém proporção de classes em ambos os conjuntos
- **Random State**: 42 (reprodutibilidade)

#### Normalização
- **StandardScaler**: Padronização com média 0 e desvio padrão 1
- **Fit em treino**: Evita data leakage
- **Transform em teste**: Aplica mesma transformação

#### Balanceamento de Classes
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- Cria amostras sintéticas de classes minoritárias
- Aplicado apenas no conjunto de treino
- Resolve desbalanceamento de classes

---

### **3. Modelos de Machine Learning**

#### **Modelo 1: Regressão Logística**
- **Tipo**: Classificação linear
- **Características**:
  - Simples e interpretável
  - Rápido para treinar
  - Serve como baseline
  - Bom para dados linearmente separáveis
- **Hiperparâmetros**:
  - `max_iter=1000`
  - `random_state=42`

#### **Modelo 2: Random Forest**
- **Tipo**: Ensemble de árvores de decisão
- **Características**:
  - Captura relações não-lineares
  - Robusto a outliers
  - Fornece importância das features
  - Generaliza bem
- **Hiperparâmetros**:
  - `n_estimators=100` (100 árvores)
  - `random_state=42`

#### **Modelo 3: XGBoost**
- **Tipo**: Gradient Boosting otimizado
- **Características**:
  - Estado-da-arte em performance
  - Otimizado para dados desbalanceados
  - Regularização integrada
  - Melhor tratamento de features
- **Hiperparâmetros**:
  - `n_estimators=100`
  - `max_depth=6`
  - `learning_rate=0.1`
  - `subsample=0.8`
  - `colsample_bytree=0.8`

---

### **4. Avaliação dos Modelos**

#### Métricas Utilizadas

**Acurácia**
- Proporção de predições corretas
- Fórmula: (TP + TN) / Total
- Intervalo: 0 a 1

**Precisão**
- De todas as predições positivas, quantas estão corretas?
- Fórmula: TP / (TP + FP)
- Importante quando falsos positivos são custosos

**Recall (Sensibilidade)**
- De todos os casos positivos, quantos foram identificados?
- Fórmula: TP / (TP + FN)
- Importante quando falsos negativos são custosos

**F1-Score**
- Média harmônica entre precisão e recall
- Fórmula: 2 × (Precisão × Recall) / (Precisão + Recall)
- Ideal para dados desbalanceados

#### Análises Adicionais
- **Matriz de Confusão**: Verdadeiros/falsos positivos/negativos
- **Relatório de Classificação**: Métricas por classe
- **Importância das Features**: Ranking de relevância
- **Comparação de Modelos**: Tabela com todas as métricas

---

### **5. Análise de Importância das Features**

#### Random Forest
- Extrai `feature_importances_` de cada árvore
- Calcula média ponderada
- Identifica features mais relevantes

#### XGBoost
- Calcula ganho (gain) de cada feature
- Considera número de splits
- Compara com Random Forest

**Resultado**: Ranking de features que mais contribuem para a predição

---

## 📈 Resultados Esperados

### Desempenho dos Modelos
- **Regressão Logística**: Baseline, acurácia ~60-70%
- **Random Forest**: Melhor generalização, acurácia ~75-85%
- **XGBoost**: Melhor performance, acurácia ~80-90%

### Features Mais Importantes
1. Idade
2. Sexo
3. Local de ocorrência
4. Escolaridade
5. Raça/Cor
6. Circunstância do óbito

### Insights Esperados
- Padrões de mortalidade por idade e sexo
- Relação entre escolaridade e causa de morte
- Diferenças por raça/cor
- Influência do local de ocorrência

---

## 🛠️ Tecnologias Utilizadas

### Bibliotecas Python
- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **scikit-learn**: Modelos de ML e pré-processamento
- **xgboost**: Gradient Boosting
- **imbalanced-learn**: SMOTE para balanceamento
- **matplotlib/seaborn**: Visualizações
- **requests**: Download de dados remoto
- **zipfile**: Extração de arquivos

### Versões Recomendadas
```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
imbalanced-learn >= 0.8.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
requests >= 2.26.0
```

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.7+
- pip ou conda
- Conexão com internet (primeira execução)

### Instalação de Dependências
```bash
pip install -r requirements.txt
```

### Execução
```bash
jupyter lab
```

### Primeira Execução
- Download do arquivo ZIP do DATASUS (~100 MB)
- Extração e processamento dos dados
- Treinamento dos 3 modelos
- Geração de visualizações
- **Tempo estimado**: 5-15 minutos

### Execuções Subsequentes
- Usa cache local do arquivo CSV
- Muito mais rápido (~5-15 minutos)
- Sem necessidade de internet

---

## 📊 Saídas Geradas

### Console Output
- Progresso do carregamento de dados
- Métricas de cada modelo
- Matriz de confusão
- Relatório de classificação
- Importância das features
- Comparação de desempenho

### Visualizações
- Gráfico de comparação de acurácia
- Matriz de confusão (heatmap)
- Top 6 features - Random Forest
- Top 6 features - XGBoost
- Arquivo PNG em alta resolução (300 DPI)

### Arquivos Salvos
- Gráficos em formato PNG
- Arquivo CSV em cache local (primeira execução)

---

## 🔍 Interpretação dos Resultados

### Acurácia Alta (>80%)
- ✅ Modelo tem boa performance
- ✅ Pode ser usado para predições
- ✅ Considerar ensemble de modelos

### Acurácia Moderada (60-80%)
- Performance aceitável
- Considerar ajuste de hiperparâmetros
- Adicionar mais features

### Acurácia Baixa (<60%)
- ✗ Performance fraca
- ✗ Revisar pré-processamento
- ✗ Considerar diferentes features

### Importância das Features
- Features com alta importância: Mais relevantes para predição
- Features com baixa importância: Podem ser removidas
- Comparação entre modelos: Validar consistência

---

## 📝 Estrutura do Desenvolvimento

### Fase 1: Exploração
- Carregamento de dados
- Análise descritiva
- Identificação de padrões

### Fase 2: Pré-processamento
- Limpeza de dados
- Tratamento de valores ausentes
- Codificação de variáveis
- Normalização

### Fase 3: Modelagem
- Divisão treino/teste
- Balanceamento de classes
- Treinamento de 3 modelos
- Avaliação inicial

### Fase 4: Avaliação
- Cálculo de métricas
- Análise de importância
- Comparação de modelos
- Geração de visualizações

### Fase 5: Interpretação
- Análise de resultados
- Identificação de insights
- Recomendações

---

## Considerações Importantes

### Qualidade dos Dados
- Dataset contém valores "Ignorado" em algumas variáveis
- Alguns registros podem ter informações incompletas
- Dados refletem padrões de 2024

### Desbalanceamento de Classes
- Algumas causas de morte são mais frequentes
- SMOTE ajuda a resolver, mas não elimina completamente
- F1-Score é mais apropriado que acurácia

### Correlação vs Causalidade
- Análises mostram correlações, não causalidade
- Features importantes não implicam relação causal
- Interpretação requer conhecimento epidemiológico

### Validação
- Sempre validar em dados não vistos
- Considerar validação cruzada
- Monitorar performance em dados novos

---

## Aprendizados

Este projeto demonstra:
- Classificação multiclasse com dados reais
- Pré-processamento completo de dados
- Balanceamento de classes (SMOTE)
- Comparação de múltiplos modelos
- Avaliação robusta com múltiplas métricas
- Análise de importância de features
- Visualizações profissionais
- Boas práticas de ML

---

## Referências

### Documentação
- [Scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Imbalanced-learn](https://imbalanced-learn.org/)

### Dados
- [DATASUS](https://datasus.saude.gov.br/)
- [OpenDataSUS](https://opendatasus.saude.gov.br/)
- [SIM - Sistema de Informações sobre Mortalidade](https://www.gov.br/saude/pt-br/acesso-a-informacao/acoes-e-programas/sistema-de-informacoes-sobre-mortalidade-sim)

### Conceitos
- [CID-10](https://www.who.int/standards/classifications/classification-of-diseases)
- [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning)
- [SMOTE](https://arxiv.org/abs/1106.1813)

---

## Licença

Este projeto utiliza dados públicos do DATASUS e segue as diretrizes de uso de dados abertos do governo brasileiro.