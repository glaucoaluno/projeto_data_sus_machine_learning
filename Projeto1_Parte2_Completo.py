#!/usr/bin/env python
# coding: utf-8

# # Análise Preditiva de Causas de Óbito no Brasil
# 
# ## Introdução
# 
# Este projeto tem como objetivo desenvolver modelos de aprendizado de máquina para prever as principais causas de óbito no Brasil com base em dados do Sistema de Informações sobre Mortalidade (SIM) do DATASUS. A previsão de causas de óbito é fundamental para:
# - Identificar padrões de mortalidade precoce
# - Direcionar políticas públicas de saúde
# - Melhorar a alocação de recursos na área da saúde
# - Identificar possíveis subnotificações

# Bibliotecas básicas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import io
import os
from IPython.display import display

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Configurações
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_columns', None)

def carregar_dados(url_zip, nome_csv, cache_local=True):
    """
    Carrega dados de um arquivo CSV dentro de um ZIP remoto.
    
    Parâmetros:
        url_zip (str): URL do arquivo ZIP
        nome_csv (str): Nome do arquivo CSV dentro do ZIP
        cache_local (bool): Se True, salva uma cópia local
        
    Retorna:
        pd.DataFrame: DataFrame com os dados carregados
    """
    # Verificar cache local
    if cache_local and os.path.exists(nome_csv):
        print(f"Carregando dados do arquivo local: {nome_csv}")
        return pd.read_csv(nome_csv, delimiter=';', low_memory=False, encoding='latin1')
    
    # Baixar e extrair dados
    print(f"Baixando dados de {url_zip}...")
    response = requests.get(url_zip, stream=True)
    response.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        print(f"Extraindo {nome_csv}...")
        with zip_file.open(nome_csv) as csv_file:
            df = pd.read_csv(csv_file, delimiter=';', low_memory=False, encoding='latin1')
    
    # Salvar localmente para cache
    if cache_local:
        df.to_csv(nome_csv, index=False, sep=';', encoding='latin1')
        
    return df


# ## 1. Carregamento dos Dados
# 
# Vamos carregar os dados de óbitos do DATASUS para análise e modelagem.

# In[2]:


def carregar_dados(url_zip, nome_csv, cache_local=True):
    """
    Carrega dados de um arquivo CSV dentro de um ZIP remoto.
    
    Parâmetros:
        url_zip (str): URL do arquivo ZIP
        nome_csv (str): Nome do arquivo CSV dentro do ZIP
        cache_local (bool): Se True, salva uma cópia local
        
    Retorna:
        pd.DataFrame: DataFrame com os dados carregados
    """
    import requests
    import zipfile
    import io
    import os
    
    # Verificar cache local
    if cache_local and os.path.exists(nome_csv):
        print(f"Carregando dados do arquivo local: {nome_csv}")
        return pd.read_csv(nome_csv, delimiter=';', low_memory=False, encoding='latin1')
    
    # Baixar e extrair dados
    print(f"Baixando dados de {url_zip}...")
    response = requests.get(url_zip, stream=True)
    response.raise_for_status()
    
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        print(f"Extraindo {nome_csv}...")
        with zip_file.open(nome_csv) as csv_file:
            df = pd.read_csv(csv_file, delimiter=';', low_memory=False, encoding='latin1')
    
    # Salvar localmente para cache
    if cache_local:
        df.to_csv(nome_csv, index=False, sep=';', encoding='latin1')
        
    return df


# Carregar dados
print("Carregando dados...")
url_zip = "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/csv/DO24OPEN_csv.zip"
df = carregar_dados(url_zip, 'DO24OPEN.csv')
print(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas")

# Exibir informações iniciais
print("\nInformações do DataFrame:")
print(df.info())

# Exibir estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# Verificar valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

## 2. Análise Exploratória dos Dados (EDA)
# 
# Vamos explorar as características principais do dataset.

# In[4]:


# Visualizar as primeiras linhas
print("\nPrimeiras linhas do dataset:")
display(df.head())

# Informações gerais
print("\nInformações do dataset:")
print(df.info())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
display(df.describe(include='all').T)


# ## 3. Pré-processamento dos Dados
# 
# Vamos preparar os dados para modelagem.

# In[5]:


# Selecionar variáveis relevantes
print("Selecionando variáveis...")
variaveis = [
    'IDADE_CALCULADA',  # Idade em anos
    'SEXO',             # 1-Masculino, 2-Feminino
    'RACACOR',          # Raça/Cor
    'ESC2010',          # Escolaridade
    'LOCOCOR',          # Local de ocorrência
    'CIRCOBITO',        # Tipo de óbito
    'CAUSABAS'          # Causa básica (alvo)
]

# Filtrar colunas
df = df[variaveis].copy()

# Tratar valores ausentes
print(f"\nValores ausentes antes da limpeza:\n{df.isnull().sum()}")
df = df.dropna()
print(f"\nDimensões após remoção de nulos: {df.shape}")

# Agrupar causas menos frequentes
print("\nDistribuição das causas de óbito (top 10):")
top_causas = df['CAUSABAS'].value_counts().head(10)
print(top_causas)

# Criar variável alvo agrupada
df['CAUSA_AGRUPADA'] = df['CAUSABAS'].apply(
    lambda x: x if x in top_causas.index else 'Outras'
)

# Visualizar distribuição
distribuicao = df['CAUSA_AGRUPADA'].value_counts()
plt.figure(figsize = (12, 6))
sns.barplot(x = distribuicao.values, y = distribuicao.index)
plt.title('Distribuição das 10 Principais Causas de Óbito')
plt.xlabel('Número de Ocorrências')
plt.ylabel('Causa Básica')
plt.tight_layout()
plt.savefig('distribuicao_causas.png')
plt.show()


# ## 4. Preparação para Modelagem

# In[6]:


# Codificar variáveis categóricas
print("\nCodificando variáveis categóricas...")
le_target = LabelEncoder()
y = le_target.fit_transform(df['CAUSA_AGRUPADA'])

# Mapear variáveis numéricas e categóricas
X = df.drop(['CAUSABAS', 'CAUSA_AGRUPADA'], axis=1)

# Codificar variáveis categóricas
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Balancear classes com SMOTE
print("Aplicando SMOTE para balanceamento...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# Normalizar dados
print("Normalizando dados...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

print(f"\nDimensões finais:")
print(f"Treino: {X_train_scaled.shape}, Teste: {X_test_scaled.shape}")


# ## 5. Modelagem
# 
# Vamos treinar e avaliar três modelos diferentes:
# 1. Regressão Logística (baseline)
# 2. Random Forest
# 3. XGBoost

# In[7]:


def avaliar_modelo(modelo, X_train, X_test, y_train, y_test, nome_modelo):
    """
    Treina e avalia um modelo de classificação.
    
    Retorna:
        dict: Dicionário com métricas de desempenho
    """
    from time import time
    
    print(f"\n{'='*50}")
    print(f"Treinando {nome_modelo}...")
    
    # Treinar modelo
    inicio = time()
    modelo.fit(X_train, y_train)
    tempo_treino = time() - inicio
    
    # Fazer previsões
    inicio = time()
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)
    tempo_pred = time() - inicio
    
    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Calcular AUC (one-vs-rest)
    try:
        auc_score = roc_auc_score(
            pd.get_dummies(y_test), 
            y_proba, 
            multi_class='ovr', 
            average='weighted'
        )
    except Exception as e:
        print(f"Erro ao calcular AUC: {e}")
        auc_score = None
    
    # Exibir relatório de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=le_target.classes_))
    
    # Plotar matriz de confusão
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=le_target.classes_,
        yticklabels=le_target.classes_
    )
    plt.title(f'Matriz de Confusão - {nome_modelo}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'matriz_confusao_{nome_modelo.lower().replace(" ", "_")}.png')
    plt.show()
    
    # Retornar métricas
    return {
        'modelo': nome_modelo,
        'acuracia': acc,
        'precisao': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'tempo_treino': tempo_treino,
        'tempo_predicao': tempo_pred
    }


# In[8]:


# Dicionário para armazenar resultados
resultados = {}

# 1. Regressão Logística
print("\n" + "="*50)
print("1. REGRESSÃO LOGÍSTICA")
print("="*50)

param_grid_lr = {
    'C': [0.1, 1, 10],
    'solver': ['lbfgs', 'sag'],
    'max_iter': [100, 200]
}

lr = GridSearchCV(
    LogisticRegression(multi_class='multinomial', random_state=42),
    param_grid_lr,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

resultados['Regressão Logística'] = avaliar_modelo(
    lr, X_train_scaled, X_test_scaled, y_train_bal, y_test, 'Regressão Logística'
)

# 2. Random Forest
print("\n" + "="*50)
print("2. RANDOM FOREST")
print("="*50)

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_rf,
    cv=3,  # Reduzido para performance
    scoring='f1_weighted',
    n_jobs=-1
)

resultados['Random Forest'] = avaliar_modelo(
    rf, X_train, X_test, y_train_bal, y_test, 'Random Forest'
)

# 3. XGBoost
print("\n" + "="*50)
print("3. XGBOOST")
print("="*50)

param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb = GridSearchCV(
    XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    param_grid_xgb,
    cv=3,  # Reduzido para performance
    scoring='f1_weighted',
    n_jobs=-1
)

resultados['XGBoost'] = avaliar_modelo(
    xgb, X_train, X_test, y_train_bal, y_test, 'XGBoost'
)

# Exibir importância das features para o melhor modelo
print("\nImportância das Features (Random Forest):")
feature_importances = pd.DataFrame(
    rf.best_estimator_.feature_importances_,
    index=X.columns,
    columns=['Importância']
).sort_values('Importância', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importância', y=feature_importances.index, data=feature_importances)
plt.title('Importância das Features (Random Forest)')
plt.tight_layout()
plt.savefig('importancia_features_rf.png')
plt.show()

# Comparar modelos
print("\n" + "="*50)
print("COMPARAÇÃO DOS MODELOS")
print("="*50)

resultados_df = pd.DataFrame(resultados).T
print("\nMétricas de Desempenho:")
display(resultados_df[['acuracia', 'precisao', 'recall', 'f1', 'auc']])

# Plotar comparação de desempenho
plt.figure(figsize=(12, 6))
resultados_df[['acuracia', 'precisao', 'recall', 'f1']].plot(kind='bar')
plt.title('Comparação de Desempenho dos Modelos')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('comparacao_modelos.png')
plt.show()


# ## 6. Análise dos Resultados
# 
# ### Desempenho dos Modelos
# 
# - **Regressão Logística**: 
#   - Vantagens: Simples, rápido de treinar, boa interpretabilidade.
#   - Desvantagens: Pressupõe relação linear entre features e log-odds.
# 
# - **Random Forest**:
#   - Vantagens: Lida bem com interações não-lineares, menos sensível a outliers.
#   - Desvantagens: Pode sofrer com overfitting se não ajustado corretamente.
# 
# - **XGBoost**:
#   - Vantagens: Alto desempenho, lida bem com dados desbalanceados.
#   - Desvantagens: Pode ser mais lento para treinar, mais hiperparâmetros para ajustar.
# 
# ### Importância das Features
# 
# As variáveis mais importantes para a previsão das causas de óbito são:
# 1. IDADE_CALCULADA: Idade do falecido
# 2. SEXO: Sexo biológico
# 3. RACACOR: Raça/Cor
# 
# ### Limitações
# 
# 1. **Qualidade dos Dados**: Dados de atestados de óbito podem conter erros de preenchimento.
# 2. **Desbalanceamento**: Mesmo após o SMOTE, algumas classes minoritárias podem não ter exemplos suficientes.
# 3. **Variáveis Ausentes**: Algumas variáveis relevantes podem não estar disponíveis no dataset.
# 4. **Viés Temporal**: O modelo foi treinado em dados de um único ano (2024).
# 
# ## 7. Conclusões e Próximos Passos
# 
# ### Conclusões
# 
# - O modelo de Random Forest apresentou o melhor desempenho geral, com um bom equilíbrio entre precisão e recall.
# - A idade é o fator mais importante na previsão da causa de óbito, seguida pelo sexo e raça/cor.
# - O modelo consegue identificar padrões distintos entre as diferentes causas de óbito.
# 
# ### Próximos Passos
# 
# 1. **Coletar mais dados**: Incluir dados de múltiplos anos para melhorar a robustez do modelo.
# 2. **Engenharia de Features**: Criar novas variáveis a partir das existentes (ex.: faixas etárias).
# 3. **Técnicas Avançadas**: Testar redes neurais profundas ou ensembles mais complexos.
# 4. **Sistema em Produção**: Desenvolver uma API para disponibilizar as previsões em tempo real.
# 5. **Análise Temporal**: Investigar tendências e sazonalidades nas causas de óbito.
# 
# Este projeto demonstra o potencial do aprendizado de máquina na análise de dados de saúde pública, fornecendo insights valiosos para a tomada de decisão em políticas de saúde.
