import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def stacked_bar_plot_sentiment_by_lob(data, filtro_1, campo_sentimiento):
    """Crea un gráfico de barras apiladas con Plotly."""
    
    # Preparar datos con porcentajes
    df_grouped = data.groupby([filtro_1, campo_sentimiento]).size().unstack(fill_value=0)
    df_pct = df_grouped.div(df_grouped.sum(axis=1), axis=0) * 100
    
    # Crear figura
    fig = go.Figure()
    
    # Colores personalizados
    colors = px.colors.qualitative.Set3
    
    # Agregar cada sentimiento como una barra
    for idx, sentimiento in enumerate(df_pct.columns):
        fig.add_trace(go.Bar(
            name=sentimiento,
            x=df_pct.index,
            y=df_pct[sentimiento],
            text=[f'{val:.1f}%' for val in df_pct[sentimiento]],
            textposition='inside',
            textfont=dict(size=12, color='white', family='Arial Black'),
            marker_color=colors[idx % len(colors)],
            hovertemplate='<b>%{x}</b><br>' +
                         f'{sentimiento}: %{{y:.1f}}%<br>' +
                         '<extra></extra>'
        ))
    
    # Configurar layout
    fig.update_layout(
        title=dict(
            text=f'Distribución de Sentimientos por {filtro_1} (%)',
            font=dict(size=20, family='Arial Black')
        ),
        xaxis_title=dict(text=filtro_1, font=dict(size=14, family='Arial')),
        yaxis_title=dict(text='Porcentaje (%)', font=dict(size=14, family='Arial')),
        barmode='stack',
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=1200,
        font=dict(size=12),
        legend=dict(
            title=dict(text=campo_sentimiento, font=dict(size=14)),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        yaxis=dict(range=[0, 100], gridcolor='lightgray')
    )
    
    return fig

def plot_sentiment_evolution(data, campo_fecha, campo_sentimiento, sentimiento_filtro):
    """Crea un gráfico de evolución temporal con Plotly."""
    
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
    
    # Crear figura con Plotly Express
    fig = px.line(
        data_filtered,
        x=campo_fecha,
        y='counts',
        color=campo_sentimiento,
        markers=True,
        title='Evolución de Sentimientos por Día',
        labels={campo_fecha: 'Fecha', 'counts': 'Cantidad de Casos'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Mejorar estilo
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=10, line=dict(width=2, color='white'))
    )
    
    fig.update_layout(
        title=dict(font=dict(size=20, family='Arial Black')),
        xaxis_title=dict(font=dict(size=14, family='Arial')),
        yaxis_title=dict(font=dict(size=14, family='Arial')),
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=1400,
        font=dict(size=12),
        legend=dict(
            title=dict(text=campo_sentimiento, font=dict(size=14)),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )
    
    return fig