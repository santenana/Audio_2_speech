import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configurar estilo global de seaborn para mejor estética
sns.set_style("whitegrid")
sns.set_context("talk")  # Hace todo más grande y legible

def stacked_bar_plot_sentiment_by_lob(data, filtro_1, campo_sentimiento):
    """Crea un gráfico de barras apiladas mostrando el porcentaje de cada sentimiento por LOB."""
    
    # Aumentar DPI para mejor calidad
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)  # DPI aumentado de 100 (default) a 150
    
    # Preparar datos con porcentajes
    df_grouped = data.groupby([filtro_1, campo_sentimiento]).size().unstack(fill_value=0)
    df_pct = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100

    # Usar paleta de colores más atractiva
    colors = sns.color_palette("husl", n_colors=len(df_pct.columns))
    
    # Crear gráfico de barras apiladas con porcentajes
    df_pct.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='white', linewidth=1.5)

    ax.set_title(f"Distribución de Sentimientos por {filtro_1} (%)", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Porcentaje (%)", fontsize=13, fontweight='bold')
    ax.set_xlabel(filtro_1, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)

    # Agregar etiquetas de porcentaje
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='center', 
                    fontsize=10, fontweight='bold', color='white')

    plt.legend(title=campo_sentimiento, bbox_to_anchor=(1.02, 1), 
              loc="upper left", frameon=True, shadow=True, fontsize=11)
    plt.xticks(rotation=90, ha='right', fontsize=11)
    
    plt.yticks(fontsize=11)
    
    # Agregar grid para mejor lectura
    ax.yaxis.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    return fig

def plot_sentiment_evolution(data, campo_fecha, campo_sentimiento, sentimiento_filtro):
    """Crea un gráfico de evolución temporal de sentimientos."""
    
    # Aumentar DPI y tamaño
    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)
    
    data_copy = data.copy()
    data_copy[campo_fecha] = pd.to_datetime(data_copy[campo_fecha], errors='coerce')
    data_copy = data_copy.dropna(subset=[campo_fecha])
    
    data_pivot = data_copy.pivot_table(
        index=campo_fecha, 
        columns=campo_sentimiento, 
        aggfunc='size', 
        fill_value=0
    ).reset_index()
    
    data_melted = data_pivot.melt(
        id_vars=campo_fecha, 
        var_name=campo_sentimiento, 
        value_name='counts'
    )
    
    data_filtered = data_melted[data_melted[campo_sentimiento].isin(sentimiento_filtro)]
    
    # Usar paleta de colores más atractiva
    palette = sns.color_palette("husl", n_colors=len(sentimiento_filtro))
    
    sns.lineplot(
        data=data_filtered, 
        x=campo_fecha, 
        y='counts', 
        hue=campo_sentimiento, 
        marker='o', 
        linewidth=3,  # Líneas más gruesas
        markersize=8,  # Marcadores más grandes
        ax=ax,
        palette=palette,
        alpha=0.9
    )
    
    ax.set_title('Evolución de Sentimientos por Día', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel(campo_fecha, fontsize=14, fontweight='bold')
    ax.set_ylabel('Cantidad de Casos', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=90, ha='right', fontsize=12)
    
    plt.yticks(fontsize=12)
    
    # Mejorar la leyenda
    plt.legend(title=campo_sentimiento, bbox_to_anchor=(1.02, 1), 
              loc='upper left', frameon=True, shadow=True, 
              fontsize=12, title_fontsize=13)
    
    # Grid mejorado
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Fondo ligeramente gris para contraste
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    return fig