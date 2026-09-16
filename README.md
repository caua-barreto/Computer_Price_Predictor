# 💻 Computer Price Predictor — Machine Learning Relatório FINAL

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Boosting%20%26%20Linear-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Concluído-success?style=for-the-badge)


---

## 📌 Sobre o Projeto

Este repositório apresenta um projeto completo de **Machine Learning** para previsão de preços de computadores (laptops e desktops), utilizando um dataset de **80.000 registros** com **34 colunas** de especificações de hardware, design e conectividade.

O projeto está organizado em uma esteira de processamento com **4 notebooks** que cobrem desde a análise exploratória até a avaliação final dos modelos:

| Etapa | Notebook | Descrição |
|-------|----------|-----------|
| 1 | `EDA.ipynb` | Análise exploratória: distribuição do target, outliers, correlações e análise bivariada |
| 2 | `Tratamento.ipynb` | Pipeline de tratamento: encoding, engenharia de features, seleção e higienização |
| 3 | `Modelagem.ipynb` | Treinamento e otimização de modelos lineares e baseados em árvore via Optuna |
| 4 | `Avaliacao.ipynb` | Avaliação final, análise SHAP, diagnóstico de resíduos e veredito |

O projeto foi totalmente desenvolvido em um ambiente virtual isolado (`.venv`) para garantir a reprodutibilidade das análises.

---

## 🔄 Diferenciais do Projeto

### 📊 Transformação Estatística do Target

Durante a Análise Exploratória (EDA), identificou-se que a distribuição de preços (`price`) possuía **assimetria positiva de 0.96** e **curtose de 4.32**, indicando forte cauda à direita. A transformação **`log1p`** reduziu a assimetria para **-0.13** e a curtose para **-0.04**, aproximando os dados de uma distribuição normal teórica.

| Transformação | Assimetria | Curtose |
|---------------|------------|---------|
| Original | 0.96 | 4.32 |
| **Log1p** | **-0.13** | **-0.04** |
| Sqrt | 0.33 | 0.58 |

### 🔀 Bifurcação do Dataset

Os dados foram estrategicamente separados em dois DataFrames distintos após o pré-processamento:

| Dataset | Características | Aplicação |
|---------|-----------------|-----------|
| **Dataset Linear** | Conjunto enxuto com **10 features** de baixa multicolinearidade (`gpu_tier`, `ram_gb`, `cpu_base_ghz`, etc.) | Ideal para modelos lineares (Ridge, Lasso, ElasticNet) |
| **Dataset Tree** | Conjunto completo com One-Hot Encoding, Label Encoding e features de interação (**50+ features**) | Ideal para modelos não-lineares de boosting (XGBoost, LightGBM, CatBoost) |

### 🧬 Engenharia de Features Avançada

Foram criadas **12 features sintéticas** divididas em dois grupos:

**Sinergia de Hardware:**

| Feature | Fórmula | Interpretação |
|---------|---------|---------------|
| `power_index` | `cpu_tier × gpu_tier × ram_gb` | Potência bruta combinada do sistema |
| `ram_per_core` | `ram_gb / cpu_cores` | Proporção de memória por núcleo |
| `battery_density` | `battery_wh / weight_kg` | Eficiência energética por kg |
| `weight_per_inch` | `weight_kg / display_size_in` | Compacidade do dispositivo |
| `cpu_turbo_range` | `cpu_boost_ghz - cpu_base_ghz` | Faixa de turbo do processador |

**Correção de Viés ("Apple Tax" e Depreciação Temporal):**

| Feature | Interação | O que captura |
|---------|-----------|---------------|
| `apple_ram_gb` | `brand_Apple × ram_gb` | Prêmio de preço por RAM em produtos Apple |
| `apple_gpu_premium` | `brand_Apple × gpu_tier` | Prêmio de GPU Apple |
| `gpu_power_by_year` | `gpu_tier × release_year` | Evita que GPU antigo tenha preço de GPU atual |
| `cpu_power_by_year` | `cpu_cores × release_year` | Evita que CPU antigo tenha preço de CPU atual |

### 🔍 Higienização via CatBoost

Foi treinado um **CatBoost provisório** para detecção de anomalias. Linhas com resíduos (erro de previsão) superiores a **±$1.000** foram identificadas como outliers e removidas — totalizando **127 anomalias removidas** (0.16% dos dados).

### 🛡️ Otimização Bayesiana com Optuna

Diferente de buscas em grid tradicionais, foi utilizada **otimização bayesiana (TPE)** com **50 trials por modelo** e validação cruzada K-Fold (k=5), utilizando um **scorer customizado** que calcula o RMSE em dólares reais (revertendo a transformação logarítmica).

### 🧠 Interpretabilidade com SHAP

A análise SHAP revelou as features mais impactantes nas predições dos modelos de boosting, confirmando que `gpu_tier`, `power_index`, `ram_gb` e `resolution_norm` dominam as previsões.

---

## 🧠 Modelagem e Algoritmos Avaliados

Foram desenvolvidos e comparados **7 algoritmos** diferentes:

- **Modelos Lineares:** Linear Regression, Ridge (L2), Lasso (L1) e ElasticNet — otimizados via `RidgeCV`, `LassoCV` e `ElasticNetCV`.
- **Modelos Não Lineares (Boosting):** XGBoost, LightGBM e CatBoost — otimizados via **Optuna** com 50 trials cada.

---

## 📊 Resultados Comparativos

### Modelos Lineares (treinados com 10 features essenciais)

| Modelo | MAE Validação | RMSE Validação | R² Validação | Gap (R²) |
|--------|---------------|----------------|--------------|----------|
| Linear Regression | $207.52 | $263.88 | 0.7811 | — |
| Ridge | $207.52 | $263.88 | 0.7811 | ~0% |
| Lasso | $207.51 | $263.87 | 0.7811 | ~0% |
| ElasticNet | $207.51 | $263.88 | 0.7811 | ~0% |

> Os modelos lineares apresentaram **excelente consistência** entre si, com diferenças irrelevantes nas métricas. Os coeficientes de regularização encontrados foram extremamente baixos, indicando que a regularização teve impacto mínimo. Servem como **baseline de referência** para comparação com os modelos de árvore.

### Modelos Não Lineares Otimizados (via Optuna, dataset completo)

| Modelo | MAE Treino | MAE Validação | Gap (MAE) | RMSE Validação | R² Validação | Gap (R²) |
|--------|------------|---------------|-----------|----------------|--------------|----------|
| XGBoost | $131.01 | $134.13 | +$3.12 | $173.92 | 0.9049 | 0.44% |
| LightGBM | $128.91 | $134.12 | +$5.21 | $174.00 | 0.9048 | 0.76% |
| **CatBoost** ⭐ | $130.38 | **$133.51** | +$3.13 | **$173.16** | **0.9058** | **0.46%** |

> Os três modelos de boosting otimizados apresentam **desempenhos muito próximos**, com diferenças marginais de menos de $1 no RMSE. O CatBoost levou leve vantagem em todas as métricas de validação.

---

## 🔬 Diagnósticos Visuais e Interpretabilidade

### 1. Dispersão Real vs. Predito

Os três modelos de boosting apresentam **alta consistência nas previsões**, com agrupamentos coesos em torno da linha ideal de previsão. Os maiores desvios ocorrem no 4º quartil (computadores premium acima de $3.200).

### 2. Distribuição dos Resíduos

| Modelo | Média dos Resíduos | Desvio Padrão | Assimetria | Curtose |
|--------|--------------------|---------------|------------|---------|
| XGBoost | $7.45 | $170.07 | 0.112 | 1.386 |
| LightGBM | $7.51 | $167.76 | 0.128 | 1.171 |
| CatBoost | $7.57 | $169.38 | 0.129 | 1.125 |

> Resíduos com **baixa assimetria (~0.13)** e centrados próximo de zero, indicando ausência de viés sistemático.

### 3. QQ Plot

Os três modelos apresentam **boa aderência à normalidade** na faixa central dos quantis (-3 a +3). Desvios aparecem apenas nas caudas extremas — computadores muito baratos (<$800) e muito caros (>$3.200) — um comportamento esperado dado que promoções e lançamentos não são capturáveis apenas pelas specs de hardware.

### 4. Análise SHAP — Features Mais Relevantes

As features mais impactantes nas predições dos modelos de boosting:

| Feature | Impacto |
|---------|---------|
| **power_index** | Potência combinada do sistema (cpu × gpu × ram) |
| **gpu_tier** | Tier da GPU — feature mais correlacionada com preço (r = 0.77) |
| **ram_gb** | Quantidade de memória RAM |
| **resolution_norm** | Qualidade da tela normalizada |
| **os_macOS** | Indicador do ecossistema Apple |

### 5. Coeficientes do Modelo Linear (Lasso)

| Feature | Coeficiente |
|---------|-------------|
| gpu_tier | +0.562 |
| cpu_base_ghz | +0.560 |
| device_type_Laptop | +0.290 |
| resolution_norm | +0.192 |
| storage_gb | +0.142 |
| brand_Apple | +0.140 |

---

## 🏆 Veredito: Melhor Modelo

O **CatBoost Otimizado** consolidou-se como o modelo mais maduro e confiável para este projeto:

- Entregou o **menor MAE** em validação: **$133.51**
- Atingiu o **menor RMSE**: **$173.16**
- Obteve o **maior R²**: **0.9058** (explicando 90.58% da variação dos preços)
- Apresentou **gap controlado** entre treino e validação (MAE: +$3.13, RMSE: +$4.31), indicando **baixo overfitting**
- Resíduos **simétricos e centrados em zero** (média: $7.57)

### Melhores Hiperparâmetros (CatBoost — Optuna)

| Parâmetro | Valor |
|-----------|-------|
| iterations | 849 |
| learning_rate | 0.0593 |
| depth | 5 |
| l2_leaf_reg | 9.819 |
| subsample | 0.720 |

---

## 📁 Estrutura do Repositório

```
Computer_Price_Predictor/
├── Arquivos/
│   └── computer_prices_train_80.csv    # Dataset original (80k linhas × 34 colunas)
├── csv_gerados/
│   ├── linear.csv                      # Dataset tratado para modelos lineares
│   └── tree.csv                        # Dataset tratado para modelos de árvore
├── EDA.ipynb                           # Análise Exploratória de Dados
├── Tratamento.ipynb                    # Pipeline de tratamento e engenharia de features
├── Modelagem.ipynb                     # Treinamento e otimização de modelos
├── Avaliacao.ipynb                     # Avaliação final e diagnósticos
├── utils.py                            # Funções auxiliares (plotagens, métricas, etc.)
├── requirements.txt                    # Dependências do projeto
├── .gitignore                          # Arquivos ignorados pelo git
└── README.md                           # Este arquivo
```

---

## 🚀 Como Executar Localmente

### (1) Clone este repositório:

```bash
git clone https://github.com/caua-barreto/Computer_Price_Predictor.git
cd Computer_Price_Predictor
```

### (2) Crie um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### (3) Instale as dependências:

```bash
pip install -r requirements.txt
```

### (4) Execute os notebooks na ordem:

1. `EDA.ipynb` — Análise Exploratória
2. `Tratamento.ipynb` — Pipeline de Tratamento
3. `Modelagem.ipynb` — Treinamento e Otimização
4. `Avaliacao.ipynb` — Avaliação Final

---

## 🛠️ Tecnologias Utilizadas

- **Python 3.12**
- **Pandas** / **NumPy** — Manipulação de dados
- **Scikit-learn** — Modelos lineares, métricas e validação cruzada
- **XGBoost** / **LightGBM** / **CatBoost** — Modelos de boosting
- **Optuna** — Otimização bayesiana de hiperparâmetros
- **SHAP** — Interpretabilidade dos modelos
- **Matplotlib** / **Seaborn** — Visualização de dados
- **SciPy** — Análises estatísticas

---

*Desenvolvido por [Cauã Barreto](https://github.com/caua-barreto)*
Desenvolvido por Cauã Barreto - Junior Data Scientist
