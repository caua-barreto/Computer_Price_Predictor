import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from IPython.display import display
from matplotlib import ticker
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Union


def histograma_simples(
    df: pd.DataFrame, 
    coluna: str = 'price', 
    bins: Union[int, str] = 'sturges',
    titulo: str = None,
    xlabel: str = None,
    ylabel: str = 'Frequência',
    color: str = 'steelblue',
    figsize: tuple = (10, 6)
):
    """
    Gera um gráfico combinado com Boxplot (topo) e Histograma com KDE (base),
    destacando as linhas de Média, Mediana e Moda.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str, default='price'
        Nome da coluna numérica a ser analisada.
    bins : int ou str, default='sturges'
        Número de compartimentos do histograma ou método de cálculo (ex: 30, 'auto', 'sturges').
    titulo : str, optional
        Título principal do gráfico.
    xlabel : str, optional
        Rótulo do eixo X. Se None, usa o nome da coluna.
    ylabel : str, default='Frequência'
        Rótulo do eixo Y.
    color : str, default='steelblue'
        Cor base para o boxplot e o histograma.
    figsize : tuple, default=(10, 6)
        Dimensões da figura (largura, altura).
    """
    # 1. Extração dos dados e medidas de tendência central
    dados = df[coluna].dropna()
    media = dados.mean()
    mediana = dados.median()
    
    # Tratamento caso a coluna não tenha moda definida
    modas = dados.mode()
    moda = modas[0] if not modas.empty else None

    # 2. Configuração dos subplots compartilhando o eixo X
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, 
        sharex=True, 
        gridspec_kw={"height_ratios": (1, 4)}, 
        figsize=figsize
    )
    
    # Adiciona o título principal, se fornecido
    if titulo:
        fig.suptitle(titulo, fontsize=14, fontweight='bold')

    # 3. Boxplot Superior (usando a cor personalizada)
    sns.boxplot(x=dados, ax=ax_box, color=color, fliersize=4)
    ax_box.axvline(media, color='darkorange', linestyle='--', linewidth=1.5)
    ax_box.axvline(mediana, color='forestgreen', linestyle='--', linewidth=1.5)
    if moda is not None:
        ax_box.axvline(moda, color='firebrick', linestyle='--', linewidth=1.5)
    ax_box.set(xlabel='')
    ax_box.grid(True, linestyle='--', alpha=0.7)

    # 4. Histograma com KDE Inferior (usando bins e cor personalizados)
    sns.histplot(
        x=dados, 
        bins=bins, 
        kde=True, 
        ax=ax_hist, 
        color=color, 
        edgecolor='black', 
        alpha=0.6
    )

    # Adição das linhas verticais estatísticas no histograma
    ax_hist.axvline(media, color='darkorange', linestyle='--', linewidth=2, label=f'Média: {media:.2f}')
    ax_hist.axvline(mediana, color='forestgreen', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    if moda is not None:
        ax_hist.axvline(moda, color='firebrick', linestyle='--', linewidth=2, label=f'Moda: {moda:.2f}')

    # 5. Formatação e Layout (usando os rótulos personalizados)
    ax_hist.legend(frameon=True, fontsize=10)
    
    # Define o xlabel como o fornecido, ou usa o nome da coluna se for None
    rotulo_x = xlabel if xlabel is not None else coluna
    ax_hist.set_xlabel(rotulo_x, fontsize=12)
    
    ax_hist.set_ylabel(ylabel, fontsize=12)
    ax_hist.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    
    # Ajusta o topo do gráfico para não sobrepor o título, caso exista
    if titulo:
        plt.subplots_adjust(top=0.92)
        
    plt.show()


def resumo_estatistico(df: pd.DataFrame, coluna: str = 'price', monetario: bool = True) -> pd.DataFrame:
    """
    Calcula as principais estatísticas descritivas de uma variável numérica,
    incluindo quartis, IQR, assimetria e curtose.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str, default='price'
        Nome da coluna numérica a ser analisada.
    monetario : bool, default=True
        Se True, formata os valores numéricos com o prefixo '$'.
        Se False, formata com duas casas decimais simples.

    Retorna:
    -------
    pd.DataFrame
        Tabela com as métricas e seus respectivos valores.
    """
    serie = df[coluna].dropna()
    
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    
    fmt = lambda v: f"${v:,.2f}" if monetario else f"{v:,.2f}"
    
    metricas = {
        'Métrica': [
            'Média', 
            'Mediana', 
            'Desvio Padrão', 
            'Mínimo', 
            '1º Quartil (Q1 - 25%)', 
            '3º Quartil (Q3 - 75%)', 
            'Intervalo Interquartil (IQR)', 
            'Máximo',
            'Assimetria (Skewness)', 
            'Curtose (Kurtosis)'
        ],
        'Valor': [
            fmt(serie.mean()),
            fmt(serie.median()),
            fmt(serie.std()),
            fmt(serie.min()),
            fmt(q1),
            fmt(q3),
            fmt(iqr),
            fmt(serie.max()),
            f"{serie.skew():.2f}",
            f"{serie.kurt():.2f}"
        ]
    }
    
    return pd.DataFrame(metricas)

def plot_distribuicao(
    df: pd.DataFrame, 
    coluna: str = 'price', 
    transformacao: str = 'original', 
    cor: str = None, 
    bins: int = 50, 
    figsize: tuple = (10, 6)
) -> pd.DataFrame:
    """
    Plota a distribuição de uma variável com KDE e ajuste normal teórico,
    e retorna as principais métricas estatísticas da distribuição.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    coluna : str, default='price'
        Coluna numérica a ser analisada.
    transformacao : str, default='original'
        Tipo de transformação: 'original', 'sqrt', 'log', 'log1p', 'cbrt'.
    cor : str, optional
        Cor do histograma. Aceita nomes (ex: 'royalblue') ou Hex (ex: '#FF5733').
    bins : int, default=50
        Número de bins do histograma.
    figsize : tuple, default=(10, 6)
        Dimensões da figura (largura, altura).
        
    Retorna:
    -------
    pd.DataFrame
        DataFrame contendo Média, Mediana, Desvio Padrão, Assimetria e Curtose.
    """
    serie = df[coluna].dropna()

    padrao_cores = {
        'original': 'royalblue',
        'sqrt': 'darkorange',
        'log': 'forestgreen',
        'log1p': 'seagreen',
        'cbrt': 'purple'
    }

    t = (transformacao or 'original').lower()

    # Definição dos dados transformados e cores
    if t in ('original', 'none', 'nenhuma'):
        dados = serie
        label_trans = 'Original'
        cor_final = cor if cor else padrao_cores['original']
    elif t == 'sqrt':
        dados = np.sqrt(serie)
        label_trans = 'Raiz Quadrada (Sqrt)'
        cor_final = cor if cor else padrao_cores['sqrt']
    elif t == 'log':
        dados = np.log(serie)
        label_trans = 'Logaritmo Natural (log)'
        cor_final = cor if cor else padrao_cores['log']
    elif t == 'log1p':
        dados = np.log1p(serie)
        label_trans = 'Logaritmo Natural (log1p)'
        cor_final = cor if cor else padrao_cores['log1p']
    elif t == 'cbrt':
        dados = np.cbrt(serie)
        label_trans = 'Raiz Cúbica (cbrt)'
        cor_final = cor if cor else padrao_cores['cbrt']
    else:
        raise ValueError(
            f"Transformação '{transformacao}' inválida. Opções: 'original', 'sqrt', 'log', 'log1p', 'cbrt'."
        )

    plt.figure(figsize=figsize)

    # 1. Histograma com densidade KDE (O parâmetro color já aceita Hex nativamente)
    sns.histplot(
        dados, 
        kde=True, 
        stat="density", 
        color=cor_final, 
        bins=bins, 
        edgecolor='black', 
        alpha=0.6, 
        label='Densidade Real (KDE)'
    )

    # 2. Ajuste da Normal Teórica
    mu, std = stats.norm.fit(dados)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = stats.norm.pdf(x, mu, std)

    plt.plot(
        x, p, 'r--', linewidth=2, 
        label=f'Ajuste Normal Teórico\n($\\mu$={mu:,.2f}, $\\sigma$={std:,.2f})'
    )

    # 3. Formatação Visual
    plt.title(f'Distribuição de {coluna} ({label_trans}) com Ajuste Normal', fontsize=13, fontweight='bold')
    plt.xlabel(f'{coluna} ({label_trans})', fontsize=11)
    plt.ylabel('Densidade', fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(axis='y', alpha=0.5, linestyle='--')

    plt.tight_layout()
    plt.show()

    # 4. Cálculo das Estatísticas para Retorno
    df_stats = pd.DataFrame({
        'Transformação': [label_trans],
        'Média': [dados.mean()],
        'Mediana': [dados.median()],
        'Desvio Padrão': [dados.std()],
        'Assimetria (Skew)': [dados.skew()],
        'Curtose (Kurt)': [dados.kurt()]
    })
    
    # Formatação para visualização limpa
    for col in ['Média', 'Mediana', 'Desvio Padrão', 'Assimetria (Skew)', 'Curtose (Kurt)']:
        df_stats[col] = df_stats[col].apply(lambda val: f"{val:.4f}")
        
    return df_stats

def plot_boxplot_outliers(
    df: pd.DataFrame, 
    coluna: str = 'price', 
    monetario: bool = True, 
    cor: str = '#D65F5F', 
    figsize: tuple = (12, 5)
) -> dict:
    """
    Gera um boxplot horizontal destacando os limites de Tukey e outliers.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame com os dados.
    coluna : str, default='price'
        Nome da coluna numérica.
    monetario : bool, default=True
        Se True, formata o eixo X com '$' e notação 'K'.
    cor : str, default='#D65F5F'
        Cor de preenchimento do boxplot.
    figsize : tuple, default=(12, 5)
        Dimensões da figura.

    Retorna:
    -------
    dict
        Dicionário com estatísticas dos limites e contagem de outliers.
    """
    serie = df[coluna].dropna()
    
    # 1. Cálculos estatísticos (Regra IQR de Tukey)
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_superior = q3 + 1.5 * iqr
    limite_inferior = q1 - 1.5 * iqr
    
    outliers_sup = serie[serie > limite_superior]
    outliers_inf = serie[serie < limite_inferior]
    total_outliers = len(outliers_sup) + len(outliers_inf)

    # 2. Plotagem
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        x=serie, 
        color=cor, 
        width=0.4, 
        linewidth=1.5,
        ax=ax,
        flierprops={
            'marker': 'o',
            'markerfacecolor': 'white', 
            'markeredgecolor': '#4A4A4A',
            'alpha': 0.5,
            'markersize': 5
        }
    )

    # 3. Linha e anotação do limite superior
    ax.axvline(limite_superior, color='#2B2B2B', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Deslocamento dinâmico baseado na amplitude dos dados
    offset = (serie.max() - serie.min()) * 0.015
    texto_corte = f'Limite Superior\n${limite_superior:,.2f}' if monetario else f'Limite Superior\n{limite_superior:,.2f}'
    
    ax.text(
        limite_superior + offset, -0.28, 
        texto_corte, 
        color='#2B2B2B', fontsize=10, va='center'
    )

    # 4. Formatação de eixos
    if monetario:
        formatter = ticker.FuncFormatter(lambda x, pos: f'${x/1000:,.1f}K' if abs(x) >= 1000 else f'${x:,.0f}')
        ax.xaxis.set_major_formatter(formatter)

    ax.set_title(f'Distribuição de {coluna} e Detecção de Outliers', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(f'{coluna} ($)' if monetario else coluna, fontsize=12, labelpad=10)
    ax.set_yticks([])
    sns.despine(left=True)

    plt.tight_layout()
    plt.show()

    # 5. Saída de diagnóstico
    print(f"--- Diagnóstico de Outliers: {coluna} ---")
    print(f"Limite Superior (Q3 + 1.5*IQR): {limite_superior:,.2f}")
    print(f"Limite Inferior (Q1 - 1.5*IQR): {limite_inferior:,.2f}")
    print(f"Outliers Superiores: {len(outliers_sup)} registros")
    print(f"Outliers Inferiores: {len(outliers_inf)} registros")
    print(f"Total de Outliers: {total_outliers} ({(total_outliers / len(serie)) * 100:.2f}%)")

    return {
        'limite_inferior': limite_inferior,
        'limite_superior': limite_superior,
        'qtd_outliers_sup': len(outliers_sup),
        'qtd_outliers_inf': len(outliers_inf),
        'total_outliers': total_outliers
    }

def plot_correlacao_alvo(
    df: pd.DataFrame, 
    alvo: str = 'price',
    top_n: int = 10,
    metodo: str = 'pearson',
    cor: str = 'coolwarm',
    figsize: tuple = (4, 8)
) -> pd.DataFrame:
    """
    Gera um heatmap de correlação focado APENAS na variável alvo.
    Retorna um DataFrame com TODAS as features ordenadas pela correlação.
    
    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    alvo : str, default='price'
        Nome da variável alvo.
    top_n : int, default=10
        Número de features com maior correlação para exibir no gráfico.
    metodo : str, default='pearson'
        Método estatístico: 'pearson', 'spearman' ou 'kendall'.
    cor : str, default='coolwarm'
        Paleta de cores (cmap) do heatmap. Ex: 'viridis', 'magma', 'coolwarm'.
    figsize : tuple, default=(4, 8)
        Dimensões do gráfico. Formato vertical é mais indicado.
        
    Retorna:
    -------
    pd.DataFrame
        DataFrame com TODAS as colunas numéricas ordenadas pela correlação com o alvo.
    """
    # 1. Filtra apenas colunas numéricas, ignorando IDs
    df_num = df.select_dtypes(include=[np.number]).drop(columns=['ID', 'id'], errors='ignore')
    
    if alvo not in df_num.columns:
        raise ValueError(f"A variável alvo '{alvo}' não é numérica ou não está no DataFrame.")
        
    # 2. Calcula a matriz e isola apenas a coluna do alvo
    corr = df_num.corr(method=metodo.lower())
    corr_alvo = corr[[alvo]].sort_values(by=alvo, ascending=False)
    
    # 3. Seleciona as N variáveis com maior correlação para o gráfico (incluindo o próprio alvo = 1.0)
    features_plot = corr_alvo.head(top_n + 1)
    
    # 4. Configuração e Plotagem do Gráfico
    plt.figure(figsize=figsize)
    sns.heatmap(
        features_plot,
        cmap=cor,
        annot=True,
        fmt=".3f",
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": .8},
        linewidths=.5
    )
    plt.title(f"Top {top_n} Correlações com '{alvo}'\n({metodo.capitalize()})", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel('') # Remove rótulo desnecessário no eixo Y
    plt.tight_layout()
    plt.show()
    
    # 5. Retorna o DataFrame completo
    return corr_alvo


def analisar_valores_nulos(df: pd.DataFrame, plotar: bool = True, figsize: tuple = (10, 6)) -> pd.DataFrame:
    """
    Analisa os valores nulos do DataFrame, retornando uma tabela com a quantidade 
    e o percentual de dados faltantes por coluna. Opcionalmente, plota um gráfico.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame a ser analisado.
    plotar : bool, default=True
        Se True, gera um gráfico de barras horizontal com o percentual de nulos.
    figsize : tuple, default=(10, 6)
        Dimensões do gráfico gerado.

    Retorna:
    -------
    pd.DataFrame
        Tabela contendo colunas com nulos, ordenadas da maior para a menor quantidade.
    """
    # Cálculos de contagem e percentual
    total_nulos = df.isnull().sum()
    percentual_nulos = (total_nulos / len(df)) * 100
    
    # Criando o DataFrame de resumo
    df_nulos = pd.DataFrame({
        'Total Nulos': total_nulos, 
        'Percentual (%)': percentual_nulos,
        'Tipo de Dado': df.dtypes
    })
    
    # Filtrando apenas colunas que possuem nulos e ordenando
    df_nulos = df_nulos[df_nulos['Total Nulos'] > 0].sort_values(by='Total Nulos', ascending=False)
    
    # Verifica se existem nulos antes de tentar plotar
    if df_nulos.empty:
        print("Nenhum valor nulo encontrado no DataFrame!")
        return df_nulos

    # Plotagem do gráfico
    if plotar:
        plt.figure(figsize=figsize)
        sns.set_theme(style="whitegrid")
        
        ax = sns.barplot(
            x='Percentual (%)', 
            y=df_nulos.index, 
            data=df_nulos, 
            color='#E66C37', 
            edgecolor='black',
            alpha=0.8
        )
        
        # Adicionando os rótulos nos valores das barras
        for p in ax.patches:
            width = p.get_width()
            plt.text(
                width + 0.5, 
                p.get_y() + p.get_height()/2. + 0.1, 
                f'{width:.1f}%', 
                ha="left", 
                va="center",
                fontsize=10,
                color='black'
            )
            
        plt.title('Percentual de Valores Nulos por Variável', fontsize=14, fontweight='bold')
        plt.xlabel('Percentual (%)', fontsize=12)
        plt.ylabel('Variáveis', fontsize=12)
        plt.xlim(0, max(df_nulos['Percentual (%)']) + 10) # Dá espaço para o texto na barra
        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.show()
        
    return df_nulos

def plot_dispersao_tendencia(df: pd.DataFrame, x_col: str, y_col: str = 'price', figsize: tuple = (8, 5)):
    """
    Gera um gráfico de dispersão com linha de regressão (tendência linear).
    Ideal para avaliar a relação bivariada entre duas variáveis numéricas.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    x_col : str
        Nome da variável preditora (eixo X).
    y_col : str, default='price'
        Nome da variável alvo (eixo Y).
    figsize : tuple, default=(8, 5)
        Dimensões da figura (largura, altura).
    """
    plt.figure(figsize=figsize)
    sns.regplot(
        data=df, 
        x=x_col, 
        y=y_col,
        scatter_kws={'alpha': 0.4, 'color': 'royalblue', 's': 20},
        line_kws={'color': 'firebrick', 'linewidth': 2}
    )
    plt.title(f'Relação entre {x_col} e {y_col}', fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_categoria_preco(
    df: pd.DataFrame, 
    col_cat: str, 
    col_alvo: str = 'price', 
    top_n: int = 10, 
    figsize: tuple = (12, 6)
):
    """
    Analisa a distribuição do alvo dentro das N categorias mais frequentes, 
    ordenadas de forma decrescente pela mediana.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados.
    col_cat : str
        Nome da variável categórica (eixo X).
    col_alvo : str, default='price'
        Nome da variável alvo numérica (eixo Y).
    top_n : int, default=10
        Quantidade máxima de categorias (as mais frequentes) a serem exibidas.
    figsize : tuple, default=(12, 6)
        Dimensões da figura (largura, altura).
    """
    # Filtra as Top N categorias mais frequentes para não poluir o gráfico
    top_categorias = df[col_cat].value_counts().nlargest(top_n).index
    df_filtrado = df[df[col_cat].isin(top_categorias)]
    
    # Ordena as caixas (boxes) de forma decrescente pela mediana
    ordem = df_filtrado.groupby(col_cat)[col_alvo].median().sort_values(ascending=False).index

    plt.figure(figsize=figsize)
    sns.boxplot(
        data=df_filtrado, 
        x=col_cat, 
        y=col_alvo, 
        order=ordem,
        palette="viridis",
        showfliers=False # Oculta outliers para focar na distribuição de massa central
    )
    
    plt.title(f'Distribuição de {col_alvo} por {col_cat} (Top {top_n})', fontsize=14, fontweight='bold')
    plt.xlabel(col_cat, fontsize=12)
    plt.ylabel(col_alvo, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    plt.tight_layout()
    plt.show()

def analisar_multicolinearidade_pares(df, limite=0.80):
    # Calcula a matriz de correlação (em valores absolutos)
    matriz_corr = df.select_dtypes(include=[np.number]).corr().abs()
    
    # Pega apenas o triângulo superior para evitar pares duplicados (ex: A-B e B-A)
    triangulo_sup = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
    
    # Filtra e lista os pares acima do limite
    pares_altos = [
        (linha, coluna, triangulo_sup.loc[linha, coluna]) 
        for coluna in triangulo_sup.columns 
        for linha in triangulo_sup.index 
        if triangulo_sup.loc[linha, coluna] > limite
    ]
    
    print(f"--- Pares com Correlação > {limite} ---")
    for par in sorted(pares_altos, key=lambda x: x[2], reverse=True):
        print(f"{par[0]} & {par[1]}: {par[2]:.3f}")


def aplicar_feature_engineering(df):
    """
    Cria features derivadas que capturam sinergia entre componentes de hardware
    e características físicas de mobilidade.

    Features geradas:
    - power_index: cpu_tier * gpu_tier * ram_gb
    - ram_per_core: ram_gb / cpu_cores
    - battery_density: battery_wh / weight_kg
    - weight_per_inch: weight_kg / display_size_in
    - cpu_turbo_range: cpu_boost_ghz - cpu_base_ghz

    Restrição: usa colunas removidas por remover_colunas_colinares
    (cpu_tier, battery_wh, weight_kg, cpu_boost_ghz), portanto precisa
    rodar ANTES da remoção.
    """
    df_copia = df.copy()
    epsilon = 1e-5

    df_copia['power_index'] = df_copia['cpu_tier'] * df_copia['gpu_tier'] * df_copia['ram_gb']
    df_copia['ram_per_core'] = df_copia['ram_gb'] / (df_copia['cpu_cores'] + epsilon)
    df_copia['battery_density'] = df_copia['battery_wh'] / (df_copia['weight_kg'] + epsilon)
    df_copia['weight_per_inch'] = df_copia['weight_kg'] / (df_copia['display_size_in'] + epsilon)
    df_copia['cpu_turbo_range'] = df_copia['cpu_boost_ghz'] - df_copia['cpu_base_ghz']

    return df_copia


def correcao_feature(df):
    """
    Cria features de interação que corrigem vieses sistemáticos:
    - Apple Tax: interações com brand_Apple (RAM, CPU, GPU, resolução)
    - Depreciação temporal: gpu/cpu power por ano de lançamento
    - Fator desktop compacto: SFF que não é laptop
    """
    df_copia = df.copy()

    # Features da cauda direita (Apple Tax)
    df_copia['apple_ram_gb'] = df_copia['brand_Apple'] * df_copia['ram_gb']
    df_copia['apple_cpu_cores'] = df_copia['brand_Apple'] * df_copia['cpu_cores']
    df_copia['apple_gpu_premium'] = df_copia['brand_Apple'] * df_copia['gpu_tier']
    df_copia['apple_resolution'] = df_copia['brand_Apple'] * df_copia['resolution_norm']

    # Features da cauda esquerda (depreciação temporal)
    df_copia['gpu_power_by_year'] = df_copia['gpu_tier'] * df_copia['release_year']
    df_copia['cpu_power_by_year'] = df_copia['cpu_cores'] * df_copia['release_year']
    df_copia['is_compact_desktop'] = df_copia['form_factor_SFF'] * (1 - df_copia['device_type_Laptop'])

    return df_copia


def pipeline_tratamento(df):
    """
    Pipeline completo de tratamento de dados para modelos baseados em árvore.
    Aplica todas as etapas de tratamento na ordem correta de dependências.

    Ordem de execução:
    1. tratar_variaveis_categoricas — encodings + resolution_norm
    2. aplicar_feature_engineering — usa colunas brutas (cpu_tier, battery_wh, etc.)
    3. correcao_feature — precisa de resolution_norm e form_factor_SFF
    4. normalizar_variaveis_numericas — escalona variáveis contínuas
    5. remover_colunas_colineares — remove redundâncias por último

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame bruto com as colunas originais do dataset.

    Retorna
    -------
    pd.DataFrame
        DataFrame tratado, pronto para modelos baseados em árvore (CatBoost, XGBoost, etc.).
    """
    df_tratado = df.copy()

    if 'price' in df_tratado.columns and 'price_log' not in df_tratado.columns:
        df_tratado['price_log'] = np.log1p(df_tratado['price'])

    df_tratado = tratar_variaveis_categoricas(df_tratado)
    df_tratado = aplicar_feature_engineering(df_tratado)
    df_tratado = correcao_feature(df_tratado)
    df_tratado = normalizar_variaveis_numericas(df_tratado)
    df_tratado = remover_colunas_colineares(df_tratado)

    print(f"=== Pipeline concluído ===")
    print(f"  {df_tratado.shape[0]} linhas × {df_tratado.shape[1]} colunas")

    return df_tratado


def reverter_escala(valores, transformacao):
    """Reverte a escala da variável alvo para US$."""
    if transformacao is None or transformacao == 'nenhum': return valores
    elif transformacao == 'log1p': return np.expm1(valores)
    elif transformacao == 'log': return np.exp(valores)
    elif transformacao == 'sqrt': return np.square(valores)
    elif transformacao == 'cbrt': return np.power(valores, 3)
    else: raise ValueError("Transformação inválida.")

def avaliar_modelo(modelo, X, y, nome_modelo, cv=5, transformacao='log1p'):
    """Aplica K-Fold CV, monitora overfitting e calcula IC 95% corrigido em US$."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Listas duplas (Treino e Validação)
    maes_val, rmses_val, r2s_val = [], [], []
    maes_train, rmses_train, r2s_train = [], [], []
    
    X_np, y_np = np.array(X), np.array(y)
    
    for train_index, val_index in kf.split(X_np):
        X_fold_train, X_fold_val = X_np[train_index], X_np[val_index]
        y_fold_train, y_fold_val = y_np[train_index], y_np[val_index]
        
        modelo.fit(X_fold_train, y_fold_train)
        
        # Previsões
        pred_fold_val = modelo.predict(X_fold_val)
        pred_fold_train = modelo.predict(X_fold_train)
        
        # Reverte a transformação para Dólares
        y_val_real = reverter_escala(y_fold_val, transformacao)
        pred_val_real = reverter_escala(pred_fold_val, transformacao)
        y_train_real = reverter_escala(y_fold_train, transformacao)
        pred_train_real = reverter_escala(pred_fold_train, transformacao)
        
        # Armazena métricas de Validação
        maes_val.append(mean_absolute_error(y_val_real, pred_val_real))
        rmses_val.append(np.sqrt(mean_squared_error(y_val_real, pred_val_real)))
        r2s_val.append(r2_score(y_val_real, pred_val_real))
        
        # Armazena métricas de Treino
        maes_train.append(mean_absolute_error(y_train_real, pred_train_real))
        rmses_train.append(np.sqrt(mean_squared_error(y_train_real, pred_train_real)))
        r2s_train.append(r2_score(y_train_real, pred_train_real))
        
    # Correção Nadeau-Bengio
    def calcular_ic(metricas):
        media = np.mean(metricas)
        variancia_amostral = np.var(metricas, ddof=1)
        prop_teste = 1 / cv
        prop_treino = 1 - prop_teste
        correcao = (1 / cv) + (prop_teste / prop_treino)
        erro_padrao_corrigido = np.sqrt(variancia_amostral * correcao)
        ic_inf, ic_sup = stats.t.interval(confidence=0.95, df=cv-1, loc=media, scale=erro_padrao_corrigido)
        return media, ic_inf, ic_sup

    # Extrai limites para Validação
    media_mae_val, mae_inf, mae_sup = calcular_ic(maes_val)
    media_rmse_val, rmse_inf, rmse_sup = calcular_ic(rmses_val)
    media_r2_val, r2_inf, r2_sup = calcular_ic(r2s_val)
    
    # Extrai médias de Treino
    media_mae_train = np.mean(maes_train)
    media_rmse_train = np.mean(rmses_train)
    media_r2_train = np.mean(r2s_train)
    
    # Margens de Erro (Validação)
    erro_mae  = abs(media_mae_val - mae_inf)
    erro_rmse = abs(media_rmse_val - rmse_inf)
    erro_r2   = abs(media_r2_val - r2_inf)

    # Gaps de Overfitting (Validação - Treino)
    gap_mae  = media_mae_val - media_mae_train
    gap_rmse = media_rmse_val - media_rmse_train
    gap_r2   = media_r2_val - media_r2_train

    # Formatação de Sinais para o Gap
    fmt_gap_mae  = f"+${gap_mae:>7.2f}" if gap_mae >= 0 else f"-${abs(gap_mae):>7.2f}"
    fmt_gap_rmse = f"+${gap_rmse:>7.2f}" if gap_rmse >= 0 else f"-${abs(gap_rmse):>7.2f}"
    fmt_gap_r2   = f"{gap_r2:>+9.4f}"

    # Exibição Estruturada Expandida
    print(f"\n{'=' * 92}")
    print(f" {nome_modelo.upper()} (K-Fold CV = {cv})")
    print(f"{'-' * 92}")
    print(f"  {'Métrica':<8} | {'Treino':<11} | {'Validação':<11} | {'Gap (V-T)':<10} | {'IC 95% (Validação)':<23} | {'Margem (±)':<10}")
    print(f"  {'-'*8} | {'-'*11} | {'-'*11} | {'-'*10} | {'-'*23} | {'-'*10}")
    print(f"  {'MAE':<8} | ${media_mae_train:>8.2f} | ${media_mae_val:>8.2f} | {fmt_gap_mae:<10} | [${mae_inf:>8.2f}, ${mae_sup:>8.2f}] | ±${erro_mae:>7.2f}")
    print(f"  {'RMSE':<8} | ${media_rmse_train:>8.2f} | ${media_rmse_val:>8.2f} | {fmt_gap_rmse:<10} | [${rmse_inf:>8.2f}, ${rmse_sup:>8.2f}] | ±${erro_rmse:>7.2f}")
    print(f"  {'R²':<8} |  {media_r2_train:>8.4f} |  {media_r2_val:>8.4f} | {fmt_gap_r2:<10} | [ {r2_inf:>8.4f},  {r2_sup:>8.4f}] |  ±{erro_r2:>7.4f}")
    print(f"{'=' * 92}\n")
    
    modelo.fit(X, y)
    return modelo

# =============================================================================
# PRÉ-PROCESSAMENTO DE DADOS
# =============================================================================

def tratar_variaveis_categoricas(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica Label Encoding em variáveis ordinais (wifi, storage_type, display_type),
    transforma resolution em pixels normalizados, remove colunas de alta cardinalidade
    e aplica One-Hot Encoding nas categóricas nominais restantes.

    Parâmetros:
    ----------
    df_original : pd.DataFrame
        DataFrame com as colunas categóricas originais.

    Retorna:
    -------
    pd.DataFrame
        DataFrame com todas as variáveis categóricas tratadas.
    """
    df = df_original.copy()

    mapa_wifi = {'Wi-Fi 5': 1, 'Wi-Fi 6': 2, 'Wi-Fi 6E': 3, 'Wi-Fi 7': 4}
    mapa_storage = {'HDD': 1, 'Hybrid': 2, 'SSD': 3, 'NVMe': 4}
    mapa_display = {'LED': 1, 'VA': 2, 'IPS': 3, 'QLED': 4, 'Mini-LED': 5, 'OLED': 6}

    df['wifi'] = df['wifi'].map(mapa_wifi)
    df['storage_type'] = df['storage_type'].map(mapa_storage)
    df['display_type'] = df['display_type'].map(mapa_display)

    if 'resolution' in df.columns:
        dimensoes = df['resolution'].str.split('x', expand=True).astype(float)
        df['total_pixels'] = dimensoes[0] * dimensoes[1]
        scaler = MinMaxScaler()
        df['resolution_norm'] = scaler.fit_transform(df[['total_pixels']])
        df = df.drop(columns=['resolution', 'total_pixels'])

    colunas_remover = ['model', 'cpu_model', 'gpu_model', 'ID']
    df = df.drop(columns=[c for c in colunas_remover if c in df.columns], errors='ignore')

    cols_categoricas = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=cols_categoricas, drop_first=True)

    cols_booleanas = df.select_dtypes(include=['bool']).columns
    df[cols_booleanas] = df[cols_booleanas].astype(int)

    return df


def remover_colunas_colineares(
    df_original: pd.DataFrame,
    colunas_extra: list = None
) -> pd.DataFrame:
    """
    Remove colunas com alta colinearidade identificadas na EDA.

    Parâmetros:
    ----------
    df_original : pd.DataFrame
        DataFrame tratado.
    colunas_extra : list, optional
        Colunas adicionais a serem removidas.

    Retorna:
    -------
    pd.DataFrame
        DataFrame sem as colunas colineares.
    """
    colunas_colineares = [
        'price',
        'cpu_boost_ghz',
        'cpu_threads',
        'cpu_tier',
        'battery_wh',
        'psu_watts',
        'weight_kg'
    ]

    if colunas_extra:
        colunas_colineares.extend(colunas_extra)

    return df_original.drop(columns=colunas_colineares, errors='ignore')


def normalizar_variaveis_numericas(
    df_original: pd.DataFrame,
    colunas_excluidas: list = None
) -> pd.DataFrame:
    """
    Aplica MinMaxScaler nas variáveis numéricas contínuas, preservando
    a variável alvo, colunas já normalizadas e colunas binárias (One-Hot).

    Parâmetros:
    ----------
    df_original : pd.DataFrame
        DataFrame com variáveis numéricas.
    colunas_excluidas : list, optional
        Colunas adicionais a serem excluídas da normalização.

    Retorna:
    -------
    pd.DataFrame
        DataFrame com variáveis numéricas normalizadas.
    """
    df = df_original.copy()

    base_excluidas = ['price_log', 'resolution_norm']
    if colunas_excluidas:
        base_excluidas.extend(colunas_excluidas)

    colunas_binarias = [
        c for c in df.columns
        if set(df[c].dropna().unique()).issubset({0, 1})
    ]

    colunas_para_normalizar = [
        c for c in df.select_dtypes(include=['int64', 'float64', 'int32']).columns
        if c not in base_excluidas and c not in colunas_binarias
    ]

    scaler = MinMaxScaler()
    df[colunas_para_normalizar] = scaler.fit_transform(df[colunas_para_normalizar])

    return df


def preparar_dados(
    df: pd.DataFrame,
    alvo: str = 'price_log',
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Pipeline completo de preparação: trata categóricas, remove colineares,
    normaliza numéricas e faz o split treino/validação.

    Parâmetros:
    ----------
    df : pd.DataFrame
        DataFrame bruto (deve conter a coluna alvo).
    alvo : str, default='price_log'
        Nome da variável alvo.
    test_size : float, default=0.2
        Proporção dos dados para validação.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Retorna:
    -------
    tuple
        (X_train, X_val, y_train, y_val, feature_names)
    """
    df_proc = tratar_variaveis_categoricas(df)
    df_proc = remover_colunas_colineares(df_proc)
    df_proc = normalizar_variaveis_numericas(df_proc)

    y = df_proc[alvo]
    X = df_proc.drop(columns=[alvo])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, y_train, y_val, list(X.columns)


# =============================================================================
# VISUALIZAÇÃO PARA MODELAGEM
# =============================================================================

def plot_residuos(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    titulo: str = 'Análise de Resíduos',
    figsize: tuple = (14, 5)
):
    """
    Gera dois gráficos lado a lado:
    1. Resíduos vs Valores Preditos (para detectar heterocedasticidade)
    2. Q-Q Plot dos resíduos (para verificar normalidade)

    Parâmetros:
    ----------
    y_real : array-like
        Valores reais da variável alvo.
    y_pred : array-like
        Valores preditos pelo modelo.
    titulo : str, default='Análise de Resíduos'
        Título principal da figura.
    figsize : tuple, default=(14, 5)
        Dimensões da figura.
    """
    residuos = np.array(y_real) - np.array(y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.scatter(y_pred, residuos, alpha=0.3, s=10, color='steelblue', edgecolors='none')
    ax1.axhline(y=0, color='firebrick', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Valores Preditos', fontsize=11)
    ax1.set_ylabel('Resíduos', fontsize=11)
    ax1.set_title('Resíduos vs Valores Preditos', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    stats.probplot(residuos, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot dos Resíduos', fontsize=12, fontweight='bold')
    ax2.get_lines()[0].set(markerfacecolor='steelblue', markeredgecolor='steelblue', markersize=3)
    ax2.get_lines()[1].set(color='firebrick')

    fig.suptitle(titulo, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_real_vs_predito(
    y_real: np.ndarray,
    y_pred: np.ndarray,
    titulo: str = 'Valores Reais vs Preditos',
    figsize: tuple = (7, 7)
):
    """
    Gera um scatter plot dos valores reais vs preditos com linha de referência
    (predição perfeita) para avaliação visual do modelo.

    Parâmetros:
    ----------
    y_real : array-like
        Valores reais da variável alvo.
    y_pred : array-like
        Valores preditos pelo modelo.
    titulo : str, default='Valores Reais vs Preditos'
        Título do gráfico.
    figsize : tuple, default=(7, 7)
        Dimensões da figura.
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    plt.figure(figsize=figsize)
    plt.scatter(y_real, y_pred, alpha=0.2, s=10, color='royalblue', edgecolors='none')

    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predição Perfeita')

    plt.xlabel('Valores Reais', fontsize=12)
    plt.ylabel('Valores Preditos', fontsize=12)
    plt.title(titulo, fontsize=13, fontweight='bold')
    plt.legend(frameon=True, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    modelo,
    feature_names: list,
    top_n: int = 20,
    titulo: str = None,
    figsize: tuple = (10, 8)
):
    """
    Plota a importância das features de um modelo treinado (coeficientes
    para modelos lineares ou feature_importances_ para árvores).

    Parâmetros:
    ----------
    modelo : estimator
        Modelo sklearn já treinado.
    feature_names : list
        Lista com os nomes das features.
    top_n : int, default=20
        Número de features mais importantes a exibir.
    titulo : str, optional
        Título do gráfico.
    figsize : tuple, default=(10, 8)
        Dimensões da figura.
    """
    if hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_
        tipo = 'Feature Importance'
    elif hasattr(modelo, 'coef_'):
        coefs = modelo.coef_
        if coefs.ndim > 1:
            coefs = coefs[0]
        importancias = np.abs(coefs)
        tipo = '|Coeficientes|'
    else:
        raise ValueError("O modelo não possui 'feature_importances_' nem 'coef_'.")

    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importância': importancias
    }).sort_values('Importância', ascending=True).tail(top_n)

    plt.figure(figsize=figsize)
    plt.barh(df_imp['Feature'], df_imp['Importância'], color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Importância', fontsize=12)
    plt.title(
        titulo or f'Top {top_n} {tipo}',
        fontsize=13, fontweight='bold'
    )
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def comparar_modelos(
    modelos: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    reverter_log: bool = True,
    cv: int = 5
) -> pd.DataFrame:
    """
    Treina múltiplos modelos, avalia com métricas e cross-validation,
    e retorna uma tabela comparativa ordenada por MAE de validação.

    Parâmetros:
    ----------
    modelos : dict
        Dicionário {nome: modelo_sklearn}.
        Ex: {'XGBoost': XGBRegressor(), 'CatBoost': CatBoostRegressor()}
    X_train, y_train : array-like
        Dados de treino.
    X_val, y_val : array-like
        Dados de validação.
    reverter_log : bool, default=True
        Se True, aplica np.expm1 antes de calcular as métricas (escala US$).
    cv : int, default=5
        Número de folds para cross-validation.

    Retorna:
    -------
    pd.DataFrame
        Tabela comparativa com MAE, RMSE, R² e CV Score para cada modelo.
    """
    resultados = []

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        pred_train = modelo.predict(X_train)
        pred_val = modelo.predict(X_val)

        if reverter_log:
            y_tr = np.expm1(y_train)
            y_vl = np.expm1(y_val)
            pred_train = np.expm1(pred_train)
            pred_val = np.expm1(pred_val)
        else:
            y_tr = y_train
            y_vl = y_val

        mae_val = mean_absolute_error(y_vl, pred_val)
        rmse_val = np.sqrt(mean_squared_error(y_vl, pred_val))
        r2_val = r2_score(y_vl, pred_val)

        cv_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
        cv_mean = -cv_scores.mean()
        if reverter_log:
            cv_mean_display = np.expm1(cv_mean)
        else:
            cv_mean_display = cv_mean

        resultados.append({
            'Modelo': nome,
            'MAE Validação': mae_val,
            'RMSE Validação': rmse_val,
            'R² Validação': r2_val,
            f'CV MAE ({cv} folds)': cv_mean_display
        })

        print(f"  ✓ {nome} — MAE: ${mae_val:,.2f} | R²: {r2_val:.4f}")

    df_resultados = pd.DataFrame(resultados).sort_values('MAE Validação')

    fmt_cols = ['MAE Validação', 'RMSE Validação', f'CV MAE ({cv} folds)']
    for col in fmt_cols:
        df_resultados[col] = df_resultados[col].apply(lambda v: f"${v:,.2f}")
    df_resultados['R² Validação'] = df_resultados['R² Validação'].apply(lambda v: f"{v:.4f}")

    return df_resultados.reset_index(drop=True)


def pipeline_modelo_completo(
    modelo,
    X_train, y_train, X_val, y_val,
    feature_names: list = None,
    nome_modelo: str = 'Modelo',
    reverter_log: bool = True,
    plotar: bool = True
):
    """
    Pipeline completo: treina, avalia com métricas, e gera todos os
    gráficos de diagnóstico (resíduos, real vs predito, feature importance).

    Parâmetros:
    ----------
    modelo : estimator
        Modelo sklearn a ser treinado.
    X_train, y_train : array-like
        Dados de treino.
    X_val, y_val : array-like
        Dados de validação.
    feature_names : list, optional
        Nomes das features (para feature importance).
    nome_modelo : str, default='Modelo'
        Nome para exibição nos gráficos.
    reverter_log : bool, default=True
        Se True, reverte log1p para métricas em US$.
    plotar : bool, default=True
        Se True, gera os gráficos de diagnóstico.

    Retorna:
    -------
    estimator
        Modelo treinado.
    """
    modelo.fit(X_train, y_train)

    pred_train = modelo.predict(X_train)
    pred_val = modelo.predict(X_val)

    if reverter_log:
        y_tr_real = np.expm1(y_train)
        y_vl_real = np.expm1(y_val)
        pred_train_real = np.expm1(pred_train)
        pred_val_real = np.expm1(pred_val)
    else:
        y_tr_real, y_vl_real = y_train, y_val
        pred_train_real, pred_val_real = pred_train, pred_val

    mae_val = mean_absolute_error(y_vl_real, pred_val_real)
    rmse_val = np.sqrt(mean_squared_error(y_vl_real, pred_val_real))
    r2_val = r2_score(y_vl_real, pred_val_real)
    mae_train = mean_absolute_error(y_tr_real, pred_train_real)

    print(f"{'='*50}")
    print(f"  {nome_modelo}")
    print(f"{'='*50}")
    print(f"  MAE Treino:     ${mae_train:,.2f}")
    print(f"  MAE Validação:  ${mae_val:,.2f}  (Gap: ${mae_val - mae_train:,.2f})")
    print(f"  RMSE Validação: ${rmse_val:,.2f}")
    print(f"  R² Validação:   {r2_val:.4f}")
    print(f"{'='*50}\n")

    if plotar:
        plot_real_vs_predito(y_vl_real, pred_val_real, titulo=f'{nome_modelo} — Real vs Predito')
        plot_residuos(y_vl_real, pred_val_real, titulo=f'{nome_modelo} — Análise de Resíduos')
        if feature_names:
            plot_feature_importance(modelo, feature_names, titulo=f'{nome_modelo} — Feature Importance')

    return modelo


def plotar_importancia_shap(dict_shap, features, top_n=15, 
                            cor_destaque="#00451C", cor_base="#5F9E37", 
                            fundo_figura='#F8F9FA', cor_texto='#2C3E50'):
    """
    Plota o ranking de importância das variáveis usando valores SHAP.
    
    Parâmetros:
    - dict_shap: Dicionário contendo { 'Nome do Modelo': matriz_de_valores_shap }
    - features: Lista ou Index do Pandas com os nomes das colunas
    - top_n: Quantidade de variáveis para exibir no ranking
    - cor_destaque: Cor das top 3 barras (padrão: Azul forte)
    - cor_base: Cor das demais barras (padrão: Azul claro)
    - fundo_figura: Cor do fundo do gráfico
    - cor_texto: Cor dos títulos e rótulos
    """
    n_modelos = len(dict_shap)
    cols = 3 if n_modelos >= 3 else n_modelos
    rows = math.ceil(n_modelos / cols)
    
    fig_width = 8 * cols
    fig_height = 7 * rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), facecolor=fundo_figura)
    
    y_title_pos = 0.98 if rows == 1 else 0.96
    fig.suptitle(f'Análise SHAP: Top {top_n} Variáveis por Impacto Médio (|SHAP|)',
                 fontsize=22, fontweight='bold', y=y_title_pos, color=cor_texto)
    
    if n_modelos == 1:
        axes = [axes]
    elif rows > 1 or cols > 1:
        axes = axes.flatten()

    for i, (nome, shap_vals) in enumerate(dict_shap.items()):
        ax = axes[i]
        ax.set_facecolor(fundo_figura)
        
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)

        ranking_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
        top_features = [features[j] for j in ranking_idx]
        top_values = mean_abs_shap[ranking_idx]

        # Aplica as cores personalizáveis passadas na função
        cores = [cor_destaque if idx < 3 else cor_base for idx in range(len(top_features))][::-1]

        y_pos = range(len(top_features)-1, -1, -1)
        bars = ax.barh(y_pos, top_values, color=cores, edgecolor='white', height=0.7)
        
        ax.bar_label(bars, fmt='%.4f', padding=6, color=cor_texto, fontsize=10, fontweight='bold')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=11, fontweight='500', color='#333333')
        ax.set_xlim(0, max(top_values) * 1.15)
        
        ax.set_xlabel('Impacto Médio |SHAP|', fontsize=12, fontweight='bold', color='#555555', labelpad=10)
        ax.set_title(f' {nome}', fontsize=16, fontweight='bold', pad=15, color=cor_texto)
        
        ax.grid(True, axis='x', linestyle='--', alpha=0.6, color='#BDC3C7')
        ax.grid(False, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#DDDDDD')
        ax.spines['bottom'].set_color('#DDDDDD')

    if n_modelos > 1:
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88 if rows > 1 else 0.85, hspace=0.35, wspace=0.25) 
    plt.show()


def plotar_top_coeficientes(df_coefs, top_n=10, titulo='Top Coeficientes do Modelo'):
    df_c = df_coefs.copy()
    
    # 1. Isolando o Top N pelo valor absoluto
    df_c['|Coeficiente|'] = df_c['Coeficiente'].abs()
    top_features = df_c.nlargest(top_n, '|Coeficiente|').sort_values('|Coeficiente|', ascending=True)

    # 2. Plotagem do Gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_top = ['#88B04B' if c >= 0 else '#C0504D' for c in top_features['Coeficiente']]

    bars = ax.barh(top_features['Feature'], top_features['Coeficiente'], 
                   color=colors_top, edgecolor='white', height=0.7, alpha=0.85)

    # Adicionando os textos
    for bar, val in zip(bars, top_features['Coeficiente']):
        offset = 0.01 if val >= 0 else -0.01
        alinhamento = 'left' if val >= 0 else 'right'
        
        ax.text(
            val + offset, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha=alinhamento, fontsize=9, fontweight='bold', color='#333'
        )

    # --- A CORREÇÃO DE MARGEM ENTRA AQUI ---
    min_val = top_features['Coeficiente'].min()
    max_val = top_features['Coeficiente'].max()
    margem = (max_val - min_val) * 0.15 # Adiciona 15% de respiro
    
    # Garante que o limite esquerdo acomode o texto da barra negativa sem esmagar o eixo Y
    limite_esq = min(0, min_val) - margem
    limite_dir = max(0, max_val) + margem
    ax.set_xlim(limite_esq, limite_dir)
    # ---------------------------------------

    ax.axvline(x=0, color='black', linewidth=1.2, linestyle='-')
    ax.set_xlabel('Coeficiente', fontsize=12, fontweight='bold')
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

    # 4. Display dos dados formatados
    print(f"\n--- Tabela das Top {top_n} Features ---")
    df_display = top_features.sort_values('|Coeficiente|', ascending=False)[['Feature', 'Coeficiente']]
    df_display.reset_index(drop=True, inplace=True)
    display(df_display)

def plotar_real_vs_predito(modelos_dict, X, y_true_log, cores_dict=None, titulo_geral='Comparação dos Modelos — Valores Reais vs Preditos'):
    """
    Gera scatter plots comparando os preços reais vs preditos para múltiplos modelos.
    
    Parâmetros:
    - modelos_dict: Dicionário com os nomes dos modelos e os estimadores treinados.
    - X: Variáveis preditoras (ex: X_tree).
    - y_true_log: Target verdadeiro em logaritmo (ex: y_tree).
    - cores_dict: Dicionário mapeando o nome do modelo para a sua cor (ex: {'XGBoost': '#E74C3C'}).
    - titulo_geral: Título principal da figura.
    """
    y_real_usd = np.expm1(y_true_log)
    n_modelos = len(modelos_dict)
    
    fig, axes = plt.subplots(1, n_modelos, figsize=(7 * n_modelos, 6))
    if n_modelos == 1:
        axes = [axes]
        
    # Garante que o dicionário de cores exista mesmo se o usuário não passar
    if cores_dict is None:
        cores_dict = {}
        
    cores_fallback = ['#34495E', '#1ABC9C', '#D35400', '#7F8C8D', '#8E44AD']

    for idx, (nome, modelo) in enumerate(modelos_dict.items()):
        ax = axes[idx]
        
        pred_log = modelo.predict(X)
        y_pred = np.expm1(pred_log)
        
        # Puxa a cor do seu dicionário, ou usa uma reserva caso o nome não exista lá
        cor = cores_dict.get(nome, cores_fallback[idx % len(cores_fallback)])
        
        ax.scatter(y_real_usd, y_pred, alpha=0.15, s=8, color=cor, edgecolors='none')
        
        min_val = min(y_real_usd.min(), y_pred.min())
        max_val = max(y_real_usd.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Predição Perfeita')
        
        mae = mean_absolute_error(y_real_usd, y_pred)
        r2 = r2_score(y_real_usd, y_pred)
        
        ax.set_xlabel('Preço Real (US$)', fontsize=11)
        ax.set_ylabel('Preço Predito (US$)', fontsize=11)
        ax.set_title(nome, fontsize=12, fontweight='bold')
        ax.text(0.05, 0.95, f'MAE: ${mae:,.2f}\nR²: {r2:.4f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(titulo_geral, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plotar_distribuicao_residuos(modelos_dict, X, y_true_log, cores_dict=None, titulo_geral='Distribuição dos Resíduos por Modelo'):
    """
    Gera histogramas da distribuição dos resíduos para múltiplos modelos,
    além de imprimir as estatísticas de dispersão e forma (Desvio Padrão, Assimetria, Curtose).
    
    Parâmetros:
    - modelos_dict: Dicionário com os nomes dos modelos e os estimadores treinados.
    - X: Variáveis preditoras.
    - y_true_log: Target verdadeiro em logaritmo.
    - cores_dict: Dicionário mapeando o nome do modelo para a sua cor.
    - titulo_geral: Título principal da figura.
    """
    y_real_usd = np.expm1(y_true_log)
    n_modelos = len(modelos_dict)
    
    fig, axes = plt.subplots(1, n_modelos, figsize=(7 * n_modelos, 5))
    if n_modelos == 1:
        axes = [axes]
        
    if cores_dict is None:
        cores_dict = {}
        
    cores_fallback = ['#34495E', '#1ABC9C', '#D35400', '#7F8C8D', '#8E44AD']
    
    print(f"=== ESTATÍSTICAS DOS RESÍDUOS ===")

    for idx, (nome, modelo) in enumerate(modelos_dict.items()):
        ax = axes[idx]
        
        # Previsão e cálculo do resíduo (Real - Predito)
        pred_log = modelo.predict(X)
        y_pred = np.expm1(pred_log)
        residuos = y_real_usd - y_pred
        
        # Converte para Series do Pandas para usar métodos estatísticos facilitados
        residuos_s = pd.Series(residuos)
        media_res = residuos_s.mean()
        std_res = residuos_s.std()
        skew_res = residuos_s.skew()
        kurt_res = residuos_s.kurtosis()
        
        # Print das estatísticas solicitadas
        print(f"\n[{nome}]")
        print(f"  Média         : ${media_res:.2f}")
        print(f"  Desvio Padrão : ${std_res:.2f}")
        print(f"  Assimetria    : {skew_res:.3f}")
        print(f"  Curtose       : {kurt_res:.3f}")
        
        cor = cores_dict.get(nome, cores_fallback[idx % len(cores_fallback)])
        
        # Histograma com curva de densidade (KDE)
        sns.histplot(residuos_s, kde=True, bins=50, color=cor, ax=ax, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        
        # Formatação do eixo e títulos
        ax.set_xlabel('Resíduo (US$)', fontsize=11)
        ax.set_ylabel('Frequência', fontsize=11)
        ax.set_title(nome, fontsize=12, fontweight='bold')
        
        # Caixinha de texto interna no gráfico (agora com Curtose incluída)
        ax.text(0.95, 0.95, f'Média: ${media_res:,.2f}\nStd: ${std_res:,.2f}\nSkew: {skew_res:.3f}\nKurt: {kurt_res:.3f}',
                transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    print("\n" + "="*33 + "\n")
    
    fig.suptitle(titulo_geral, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plotar_qq_plot_residuos(modelos_dict, X, y_true_log, cores_dict=None, titulo_geral='QQ Plot dos Resíduos por Modelo'):
    """
    Gera gráficos QQ (Quantile-Quantile) dos resíduos para verificar a normalidade dos erros.
    
    Parâmetros:
    - modelos_dict: Dicionário com os nomes dos modelos e os estimadores treinados.
    - X: Variáveis preditoras.
    - y_true_log: Target verdadeiro em logaritmo.
    - cores_dict: Dicionário mapeando o nome do modelo para a sua cor.
    - titulo_geral: Título principal da figura.
    """
    y_real_usd = np.expm1(y_true_log)
    n_modelos = len(modelos_dict)
    
    fig, axes = plt.subplots(1, n_modelos, figsize=(7 * n_modelos, 5))
    if n_modelos == 1:
        axes = [axes]
        
    if cores_dict is None:
        cores_dict = {}
        
    cores_fallback = ['#34495E', '#1ABC9C', '#D35400', '#7F8C8D', '#8E44AD']

    for idx, (nome, modelo) in enumerate(modelos_dict.items()):
        ax = axes[idx]
        
        # Previsão e cálculo do resíduo
        pred_log = modelo.predict(X)
        y_pred = np.expm1(pred_log)
        residuos = y_real_usd - y_pred
        
        cor = cores_dict.get(nome, cores_fallback[idx % len(cores_fallback)])
        
        # Gera o QQ plot no eixo específico
        stats.probplot(residuos, dist="norm", plot=ax)
        
        # O scipy cria 2 linhas no gráfico: [0] são os pontos do resíduo, [1] é a linha de tendência ideal
        pontos = ax.get_lines()[0]
        linha_ideal = ax.get_lines()[1]
        
        # Customização visual
        pontos.set(markerfacecolor=cor, markeredgecolor=cor, markersize=3, alpha=0.5)
        linha_ideal.set(color='firebrick', linewidth=2)
        
        # Sobrescrevendo os textos gerados automaticamente pelo scipy
        ax.set_title(f'QQ Plot — {nome}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Quantis Teóricos', fontsize=11)
        ax.set_ylabel('Quantis Amostrais', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(titulo_geral, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plotar_erros_por_faixa(modelos_dict, X, y_true_log, cores_dict=None, quantis=4):
    """
    Segmenta os preços reais em quantis e avalia RMSE e MAPE por faixa de preço.
    Gera tabela estilizada e gráficos de barras comparativos.
    """
    y_real_usd = np.expm1(y_true_log)
    
    # 1. Segmentar os preços em Faixas (Quantis)
    df_analise = pd.DataFrame({'Preço Real': y_real_usd})
    df_analise['Faixa de Preço'] = pd.qcut(df_analise['Preço Real'], q=quantis, 
                                           labels=['Q1 (Mais Baratos)', 'Q2 (Intermediários)', 
                                                   'Q3 (Avançados)', 'Q4 (Premium)'])
    
    # 2. Calcular previsões e agregar os erros
    resultados = []
    
    for nome, modelo in modelos_dict.items():
        y_pred = np.expm1(modelo.predict(X))
        df_analise[f'Pred_{nome}'] = y_pred
        
        for faixa in df_analise['Faixa de Preço'].unique():
            mask = df_analise['Faixa de Preço'] == faixa
            y_real_faixa = df_analise.loc[mask, 'Preço Real']
            y_pred_faixa = df_analise.loc[mask, f'Pred_{nome}']
            
            rmse = np.sqrt(mean_squared_error(y_real_faixa, y_pred_faixa))
            mape = mean_absolute_percentage_error(y_real_faixa, y_pred_faixa) * 100
            
            resultados.append({
                'Modelo': nome,
                'Faixa de Preço': faixa,
                'RMSE ($)': rmse,
                'MAPE (%)': mape,
                'Vol. de Dados': mask.sum()
            })
            
    df_resultados = pd.DataFrame(resultados).sort_values('Faixa de Preço')
    
    # 3. Exibir a Tabela de Calor
    print("=== Tabela de Erros por Faixa de Preço ===")
    tabela_pivot_mape = df_resultados.pivot(index='Faixa de Preço', columns='Modelo', values='MAPE (%)')
    tabela_pivot_rmse = df_resultados.pivot(index='Faixa de Preço', columns='Modelo', values='RMSE ($)')
    
    display(tabela_pivot_mape.style.background_gradient(cmap='OrRd', axis=1)\
            .format("{:.2f}%").set_caption("MAPE (%) - Menor é Melhor"))
    
    display(tabela_pivot_rmse.style.background_gradient(cmap='OrRd', axis=1)\
            .format("${:.2f}").set_caption("RMSE ($) - Menor é Melhor"))
    
    # 4. Plotagem dos Gráficos
    if cores_dict is None:
        cores_dict = {'XGBoost Otimizado': '#E74C3C', 'LightGBM Otimizado': '#2E86C1', 'CatBoost Otimizado': '#27AE60'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: MAPE
    sns.barplot(data=df_resultados, x='Faixa de Preço', y='MAPE (%)', hue='Modelo', 
                palette=cores_dict, ax=axes[0], edgecolor='black', alpha=0.8)
    axes[0].set_title('Erro Percentual Médio (MAPE) por Faixa', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('MAPE (%)', fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)
    axes[0].legend(loc='upper right', fontsize=9)
    
    # Gráfico 2: RMSE
    sns.barplot(data=df_resultados, x='Faixa de Preço', y='RMSE ($)', hue='Modelo', 
                palette=cores_dict, ax=axes[1], edgecolor='black', alpha=0.8)
    axes[1].set_title('Erro Absoluto (RMSE) por Faixa', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('RMSE em Dólares ($)', fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)
    axes[1].legend(loc='upper left', fontsize=9)
    
    fig.suptitle('Desempenho dos Modelos Segmentado por Custo do Hardware', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plotar_boxplot_faixa(df, col_x='ram_gb', col_y='price', titulo=None, cor='#2E86C1'):
    """
    Plota a distribuição e a variação exata de preços para cada categoria de RAM.
    Permite customizar o título e a cor principal do gráfico.
    """
    plt.figure(figsize=(12, 6))
    
    # Boxplot base com a cor dinâmica
    sns.boxplot(data=df, x=col_x, y=col_y, color=cor, 
                flierprops={'marker': 'o', 'markerfacecolor': 'red', 'alpha': 0.5})
    
    # Lógica do título: usa o fornecido ou monta um padrão
    titulo_final = titulo if titulo else f'Variação de Preço por Quantidade de {col_x}'
    
    plt.title(titulo_final, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(col_x, fontsize=12, fontweight='bold')
    plt.ylabel(f'{col_y} (US$)', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    sns.despine()
    plt.tight_layout()
    plt.show()    