import streamlit as st
import pandas as pd
import io
from langdetect import detect
import nltk
import re
import unicodedata
import os
import graficas
import matplotlib.pyplot as plt
import seaborn as sns
# Configuración de la página - DEBE SER EL PRIMER COMANDO
st.set_page_config(
    page_title="Análisis de Sentimientos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("📊 Análisis de Sentimientos en CSAT")
st.markdown("""
Esta aplicación analiza sentimientos en textos y genera análisis detallado de sentimientos.
""")

# ============================================================================
# FUNCIONES DE ANÁLISIS
# ============================================================================

@st.cache_data
def load_word_lists():
    """Carga los diccionarios de palabras"""
    
    word_lists = {
        'polite_words': {
            "es": ["gracias", "por favor", "buen dia", "seria tan amable", "le agradezco", "disculpe", "permítame"],
            "en": ["please", "thank you", "good morning", "kindly", "appreciate", "sorry", "excuse me"]
        },
        'rude_words': {
            "es": ["maldito", "idiota", "estúpido", "imbécil", "pendejo", "culero", "pinche"],
            "en": ["stupid", "idiot", "dumb", "useless", "shit", "terrible", "lazy", "damn"]
        },
        'technical_terms': {
            "es": ["sistema", "base de datos", "protocolo", "infraestructura", "algoritmo", "interfaz", "servidor"],
            "en": ["system", "database", "protocol", "infrastructure", "algorithm", "interface", "server"]
        },
        'toxic_words': {
            "es": ["tonto", "idiota", "estúpido", "imbécil", "maldito", "horrible", "perezoso"],
            "en": ["stupid", "idiot", "dumb", "useless", "shit", "terrible", "lazy"]
        },
        'desagrado': {
            "es": ["no me gusta", "malo", "pésimo", "desagradable", "asco", "horror", "feo", "fatal", 
                   "terrible", "decepcionante", "guácala", "fuchi", "repugnante", "rechazo"],
            "en": ["dislike", "bad", "terrible", "gross", "disgusting", "awful", "nasty", "poor quality", 
                   "yuck", "revolting", "hate it", "displeased", "unpleasant"]
        },
        'frustracion': {
            "es": ["intentar", "otra vez", "de nuevo", "no funciona", "bloqueado", "harto", "impotencia", 
                   "siempre lo mismo", "fallo", "error", "espera", "inútil", "incapaz"],
            "en": ["tried", "again", "still not", "stuck", "fed up", "failure", "broken", "useless", 
                   "waiting", "blocked", "annoyed", "pointless", "keep trying"]
        },
        'gratitud': {
            "es": ["gracias", "agradezco", "buenísimo", "excelente", "amable", "bendiciones", "genial", 
                   "ayudó mucho", "recomiendo", "valoro", "perfecto"],
            "en": ["thanks", "thank you", "grateful", "appreciate", "kind", "helpful", "awesome", 
                   "perfect", "blessed", "highly recommend", "supportive"]
        },
        'indiferencia': {
            "es": ["da igual", "equis", "me da lo mismo", "como sea", "no importa", "ni fu ni fa", 
                   "meh", "ok", "está bien", "sin opinión"],
            "en": ["whatever", "doesn't matter", "don't care", "anyway", "meh", "ok", "fine", 
                   "neutral", "indifferent", "as you wish", "regardless"]
        },
        'satisfaccion': {
            "es": ["contento", "feliz", "valió la pena", "logré", "funciona", "bien", "satisfecho", 
                   "gusto", "maravilla", "increíble", "justo lo que quería"],
            "en": ["happy", "satisfied", "pleased", "works", "worth it", "glad", "great", 
                   "fulfilled", "exactly", "success", "delighted"]
        },
        'rabia_ira': {
            "es": ["enojado", "furia", "basura", "estafa", "robo", "insoportable", "maldito", 
                   "odio", "indignado", "insulto", "inservible", "peor", "asco de"],
            "en": ["angry", "mad", "hate", "scam", "shameful", "outraged", "furious", "garbage", 
                   "trash", "disgrace", "infuriating", "pissed", "worst"]
        },
        'amenazas': {
            "es": ["demandar", "denuncia", "PROFECO", "legal", "abogado", "ir de la competencia", 
                   "cancelar", "baja", "quemar", "nunca más", "redes sociales", "última vez"],
            "en": ["sue", "legal action", "lawyer", "reporting", "cancel", "switching to", 
                   "never again", "last warning", "publicly", "court", "quit"]
        }
    }
    
    return word_lists

def normalize_text(text):
    """Normaliza el texto removiendo acentos"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower()

def detect_language_safe(text):
    """Detecta el idioma del texto de forma segura"""
    try:
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 3:
            return 'es'
        lang = detect(text)
        return lang if lang in ['es', 'en'] else 'es'
    except:
        return 'es'

def count_words_in_text(text, word_list, lang):
    """Cuenta cuántas palabras de una lista específica aparecen en el texto"""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    
    text_normalized = normalize_text(text)
    words_from_dict = word_list.get(lang, [])
    
    count = 0
    for word in words_from_dict:
        word_normalized = normalize_text(word)
        pattern = r"\b" + re.escape(word_normalized) + r"\b"
        if re.search(pattern, text_normalized, re.IGNORECASE):
            count += 1
    
    return count

def analyze_sentiment_dataframe(df, text_column, word_lists=None):
    """
    Analiza sentimientos en un DataFrame.
    """

    
    # Crear una copia del DataFrame
    df_result = df.copy()
    progress_bar = st.progress(0, text="Detectando idiomas...")
    
    df_result['_lang_temp'] = df_result[text_column].apply(
        lambda x: detect_language_safe(str(x)) if pd.notna(x) else 'es'
    )
    progress_bar.progress(10, text="Idiomas detectados")

    progress_bar.progress(15, text="Inicializando columnas...")
    
    # Detectar idioma para cada fila
    df_result['_lang_temp'] = df_result[text_column].apply(
        lambda x: detect_language_safe(str(x)) if pd.notna(x) else 'es'
    )
    
    # Inicializar columna de sentimiento final como 'neutro' (string, no int)
    df_result["Sentimiento_Final"] = 'neutro'

    # Procesar cada fila
    total_rows = len(df_result)
    
    # Definir categorías de sentimiento
    sentiment_categories = ['desagrado', 'frustracion', 'gratitud', 'indiferencia', 
                           'satisfaccion', 'rabia_ira', 'amenazas']
    
    for idx, row in df_result.iterrows():
        
        if idx % max(1, total_rows // 20) == 0:
            progress = 15 + int((idx / total_rows) * 80)
            progress_bar.progress(progress, text=f"Procesando fila {idx + 1}/{total_rows}...")
        
        text = row[text_column]
        if pd.isna(text) or not isinstance(text, str):
            continue
        
        lang = row['_lang_temp']
        
        # Contar palabras en cada categoría de sentimiento
        sentiment_scores = {}
        for category in sentiment_categories:
            if category in word_lists:
                sentiment_scores[category] = count_words_in_text(text, word_lists[category], lang)
        
        # Determinar el sentimiento dominante
        if sentiment_scores:
            max_score = max(sentiment_scores.values())
            if max_score > 0:
                dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                df_result.at[idx, 'Sentimiento_Final'] = dominant_sentiment
            else:
                df_result.at[idx, 'Sentimiento_Final'] = 'neutro'
        else:
            df_result.at[idx, 'Sentimiento_Final'] = 'neutro'

    
    # Eliminar columna temporal de idioma
    df_result = df_result.drop('_lang_temp', axis=1)
    progress_bar.progress(100, text="✅ Procesamiento completado!")
    return df_result



# ============================================================================
# INTERFAZ DE STREAMLIT
# ============================================================================

# Cargar diccionarios
word_lists = load_word_lists()

# Área principal
st.header("1️⃣ Carga tu archivo")

# Upload de archivo con configuración para evitar error 403
uploaded_file = st.file_uploader(
    "Selecciona un archivo CSV o Excel (máximo 200MB)",
    type=['csv', 'xlsx', 'xls'],
    help="El archivo debe contener una columna con los textos a analizar"
)

if uploaded_file is not None:
    try:
        # Leer el archivo según su tipo
        with st.spinner("Cargando archivo..."):
            if uploaded_file.name.endswith('.csv'):
                # Intentar diferentes encodings para CSV
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
            else:
                df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Archivo cargado: {uploaded_file.name}")
        st.info(f"📊 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
        
        # Advertencia para archivos grandes
        if df.shape[0] > 5000:
            st.warning(f"⚠️ Archivo grande detectado ({df.shape[0]} filas). El procesamiento tomará tiempo.")
        
        # Mostrar preview del DataFrame
        with st.expander("👁️ Vista previa del archivo", expanded=False):
            st.dataframe(df.head(5), use_container_width=False)
        
        # Selección de columna
        st.header("2️⃣ Selecciona la columna de texto")
        
        text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Intentar detectar automáticamente columnas
        default_column = None
        for col in ['CSAT', 'csat', 'comentario', 'texto', 'text', 'comment', 'Comentario']:
            if col in text_columns:
                default_column = col
                break
        
        selected_column = st.selectbox(
            "Columna con el texto a analizar:",
            options=text_columns,
            index=text_columns.index(default_column) if default_column else 0
        )
        
        # Mostrar muestra de la columna seleccionada
        st.markdown("**Muestra de textos:**")
        sample_texts = df[selected_column].dropna().head(3).tolist()
        for i, text in enumerate(sample_texts, 1):
            st.text(f"{i}. {str(text)[:150]}{'...' if len(str(text)) > 150 else ''}")
        
        # Botón para procesar
        st.header("3️⃣ Procesar análisis")
        
        
        if st.button("Analizar Sentimientos", type="primary", use_container_width=True):
            try:
                # Realizar el análisis
                df_resultado = analyze_sentiment_dataframe(
                    df, 
                    text_column=selected_column,
                    word_lists=word_lists
                )
                
                df_resultado = df_resultado[(df_resultado['Sentimiento_Final'] != 'neutro') & (df_resultado['Sentimiento_Final'] != 0)]
                
                # Guardar en session_state
                st.session_state['df_resultado'] = df_resultado
                st.session_state['selected_column'] = selected_column
                
                st.success("✅ ¡Análisis completado exitosamente!")
                
            except Exception as e:
                st.error(f"❌ Error durante el procesamiento: {str(e)}")
                st.exception(e)
        
        # Mostrar resultados si existen
        if 'df_resultado' in st.session_state:
            st.header("4️⃣ Resultados")
            
            df_resultado = st.session_state['df_resultado']
            
            # Crear pestañas para organizar los resultados
            tab1, tab3, tab4, tab2 = st.tabs(["📈 Estadísticas", "📊 Distribución", "📥 Descargar", "Ver Resultados"])
            
            # PESTAÑA 1: Estadísticas Generales
            with tab1:
                st.subheader("📈 Estadísticas Generales")
                
                # Verificar si existe la columna Sentimiento_Final
                if 'Sentimiento_Final' in df_resultado.columns:
                    # Contar la frecuencia de cada sentimiento
                    sentiment_counts = df_resultado['Sentimiento_Final'].value_counts()
                    
                    # Mapeo de sentimientos a emojis y etiquetas
                    sentiment_mapping = {
                        'gratitud': ('🙏', 'Gratitud'),
                        'satisfaccion': ('😊', 'Satisfacción'),
                        'indiferencia': ('😐', 'Indiferencia'),
                        'frustracion': ('😤', 'Frustración'),
                        'desagrado': ('😠', 'Desagrado'),
                        'rabia_ira': ('😡', 'Rabia/Ira'),
                        'amenazas': ('⚠️', 'Amenazas'),
                        'neutro': ('😶', 'Neutro')
                    }
                    
                    # Obtener sentimientos únicos del DataFrame
                    sentimientos_presentes = sentiment_counts.index.tolist()
                    
                    # Filtrar solo los sentimientos que existen en los datos
                    sentimientos_a_mostrar = [s for s in sentimientos_presentes if isinstance(s, str)]
                    
                    # Calcular número de columnas necesarias
                    num_sentimientos = len(sentimientos_a_mostrar)
                    
                    if num_sentimientos > 0:
                        # Crear filas de métricas (máximo 4 por fila)
                        num_rows = (num_sentimientos + 3) // 4  # Redondear hacia arriba
                        
                        for row_idx in range(num_rows):
                            start_idx = row_idx * 4
                            end_idx = min(start_idx + 4, num_sentimientos)
                            sentimientos_fila = sentimientos_a_mostrar[start_idx:end_idx]
                            
                            cols = st.columns(len(sentimientos_fila))
                            
                            for col_idx, sentimiento in enumerate(sentimientos_fila):
                                emoji, label = sentiment_mapping.get(sentimiento, ('📊', str(sentimiento).capitalize()))
                                count = int(sentiment_counts.get(sentimiento, 0))
                                cols[col_idx].metric(f"{emoji} {label}", count)
                            
                            # Agregar separador entre filas (excepto la última)
                            if row_idx < num_rows - 1:
                                st.markdown("---")
                    
                    # Mostrar distribución en porcentajes
                    st.markdown("---")
                    st.subheader("📊 Distribución Porcentual")
                    
                    total_casos = len(df_resultado)
                    sentiment_percentages = (sentiment_counts / total_casos * 100).round(2)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Crear DataFrame para mostrar
                        dist_df = pd.DataFrame({
                            'Sentimiento': [sentiment_mapping.get(s, ('', str(s).capitalize()))[1] if isinstance(s, str) else str(s) for s in sentiment_percentages.index],
                            'Cantidad': sentiment_counts.values,
                            'Porcentaje (%)': sentiment_percentages.values
                        })
                        st.dataframe(dist_df, hide_index=True, use_container_width=True)
                    
                    with col2:
                        # Gráfico de torta
                        fig, ax = plt.subplots(figsize=(8, 8))
                        colors = plt.cm.Set3(range(len(sentiment_counts)))
                        
                        wedges, texts, autotexts = ax.pie(
                            sentiment_counts.values,
                            labels=[sentiment_mapping.get(s, ('', str(s).capitalize()))[1] if isinstance(s, str) else str(s) for s in sentiment_counts.index],
                            autopct='%1.1f%%',
                            startangle=90,
                            colors=colors,
                            textprops={'fontsize': 10}
                        )
                        
                        # Mejorar legibilidad y rotar según dirección del slice
                        for i, (autotext, wedge) in enumerate(zip(autotexts, wedges)):
                            autotext.set_color('black')
                            autotext.set_fontweight('bold')
                            
                            # Calcular el ángulo medio del slice
                            angle = (wedge.theta2 + wedge.theta1) / 2
                            
                            # Ajustar la rotación del texto para que sea paralelo al slice
                            # Si está en la mitad izquierda (90-270°), rotar 180° adicionales para que sea legible
                            if 90 < angle < 270:
                                rotation_angle = angle - 180
                            else:
                                rotation_angle = angle
                            
                            autotext.set_rotation(rotation_angle)
                        
                        ax.set_title('Distribución de Sentimientos', fontsize=14, fontweight='bold', pad=20)
                        st.pyplot(fig)
                        plt.close()
                
                else:
                    st.warning("⚠️ No se encontró la columna 'Sentimiento_Final' en los datos.")
                    st.info("Asegúrate de que la función de análisis de sentimientos se haya ejecutado correctamente.")
                        
            # PESTAÑA 2: Vista de Resultados
            with tab2:
                st.subheader("📋 Vista de Resultados")
                st.dataframe(df_resultado, use_container_width=True, height=400)
            
            # PESTAÑA 3: Distribución de Sentimientos
            with tab3:
                st.subheader("📊 Distribución de Sentimientos")

                if 'Sentimiento_Final' in df_resultado.columns:
                    # Contar la frecuencia de cada sentimiento
                    sentiment_counts = df_resultado['Sentimiento_Final'].value_counts().sort_values(ascending=False)
                    
                    # Crear DataFrame para visualización
                    sentiment_df = pd.DataFrame({
                        'Categoría': sentiment_counts.index,
                        'Total': sentiment_counts.values
                    })
                    
                    # Mapeo de sentimientos a nombres más legibles
                    sentiment_labels = {
                        'gratitud': 'Gratitud',
                        'satisfaccion': 'Satisfacción',
                        'indiferencia': 'Indiferencia',
                        'frustracion': 'Frustración',
                        'desagrado': 'Desagrado',
                        'rabia_ira': 'Rabia/Ira',
                        'amenazas': 'Amenazas',
                        'neutro': 'Neutro'
                    }
                    
                    sentiment_df['Categoría_Label'] = sentiment_df['Categoría'].apply(
                        lambda x: sentiment_labels.get(x, str(x).capitalize()) if isinstance(x, str) else str(x)
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### Gráfico de Barras")
                        # Usar la versión con labels para mejor presentación
                        chart_df = sentiment_df.set_index('Categoría_Label')['Total']
                        st.bar_chart(chart_df)
                    
                    with col2:
                        st.markdown("##### Tabla de Frecuencias")
                        # Calcular porcentajes
                        total_casos = sentiment_df['Total'].sum()
                        display_df = pd.DataFrame({
                            'Sentimiento': sentiment_df['Categoría_Label'],
                            'Cantidad': sentiment_df['Total'],
                            'Porcentaje': (sentiment_df['Total'] / total_casos * 100).round(2).astype(str) + '%'
                        })
                        
                        st.dataframe(
                            display_df,
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "Sentimiento": st.column_config.TextColumn("Sentimiento", width="medium"),
                                "Cantidad": st.column_config.NumberColumn("Cantidad", width="small"),
                                "Porcentaje": st.column_config.TextColumn("Porcentaje", width="small")
                            }
                        )
                        
                        # Mostrar total
                        st.metric("Total de Casos", total_casos)

                else:
                    st.warning("⚠️ No se encontró la columna 'Sentimiento_Final' en los datos.")

                st.divider()
                
                # ========== GRÁFICO 1: Barras Apiladas por LOB ==========
                st.subheader("📊 Distribución de Sentimientos por Categoría")
                
                # Obtener columnas disponibles
                columnas_disponibles = df_resultado.columns.tolist()
                
                col_filtro1, col_sentimiento1 = st.columns(2)
                
                with col_filtro1:
                    filtro_1 = st.selectbox(
                        "Selecciona la variable de agrupación (filtro_1):",
                        options=columnas_disponibles,
                        index=columnas_disponibles.index('LOB') if 'LOB' in columnas_disponibles else 0,
                        key='filtro_1_stacked'
                    )
                
                with col_sentimiento1:
                    campo_sentimiento_1 = st.selectbox(
                        "Selecciona el campo de sentimiento:",
                        options=columnas_disponibles,
                        index=columnas_disponibles.index('Sentimiento_Final') if 'Sentimiento_Final' in columnas_disponibles else 0,
                        key='sentimiento_stacked'
                    )
                
                if filtro_1 and campo_sentimiento_1:
                    try:
                        fig1 = graficas.stacked_bar_plot_sentiment_by_lob(
                            df_resultado, 
                            filtro_1, 
                            campo_sentimiento_1
                        )
                        st.pyplot(fig1)
                        plt.close()
                    except Exception as e:
                        st.error(f"Error al generar el gráfico: {str(e)}")
                
                st.divider()
                
                # ========== GRÁFICO 2: Evolución Temporal de Sentimientos ==========
                st.subheader("📈 Evolución Temporal de Sentimientos")
                
                col_fecha, col_sentimiento2 = st.columns(2)
                
                with col_fecha:
                    campo_fecha = st.selectbox(
                        "Selecciona el campo de fecha:",
                        options=columnas_disponibles,
                        index=columnas_disponibles.index('Fecha') if 'Fecha' in columnas_disponibles else 0,
                        key='fecha_evolution'
                    )
                
                with col_sentimiento2:
                    campo_sentimiento_2 = st.selectbox(
                        "Selecciona el campo de sentimiento:",
                        options=columnas_disponibles,
                        index=columnas_disponibles.index('Sentimiento_Final') if 'Sentimiento_Final' in columnas_disponibles else 0,
                        key='sentimiento_evolution'
                    )
                
                # Obtener valores únicos del campo de sentimiento seleccionado
                if campo_sentimiento_2:
                    sentimientos_disponibles = df_resultado[campo_sentimiento_2].dropna().unique().tolist()
                    
                    sentimiento_filtro = st.multiselect(
                        "Selecciona los sentimientos a visualizar:",
                        options=sentimientos_disponibles,
                        default=sentimientos_disponibles,
                        key='filtro_sentimientos'
                    )
                    
                    if sentimiento_filtro and campo_fecha and campo_sentimiento_2:
                        try:
                            fig2 = graficas.plot_sentiment_evolution(
                                df_resultado, 
                                campo_fecha, 
                                campo_sentimiento_2, 
                                sentimiento_filtro
                            )
                            st.pyplot(fig2)
                            plt.close()
                        except Exception as e:
                            st.error(f"Error al generar el gráfico: {str(e)}")
                    elif not sentimiento_filtro:
                        st.warning("Por favor selecciona al menos un sentimiento para visualizar.")
            
            # PESTAÑA 4: Distribución de Sentimientos
            with tab4:
                st.subheader("📥 Descargar Resultados")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Descargar CSV
                    csv = df_resultado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Descargar como CSV",
                        data=csv,
                        file_name="analisis_sentimientos.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Descargar Excel
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df_resultado.to_excel(writer, index=False, sheet_name='Análisis')
                    
                    st.download_button(
                        label="📥 Descargar como Excel",
                        data=buffer.getvalue(),
                        file_name="analisis_sentimientos.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
    
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {str(e)}")
        st.exception(e)

else:
    # Instrucciones cuando no hay archivo
    st.info("👆 Por favor, carga un archivo CSV o Excel para comenzar el análisis.")
    
    # Ejemplo de formato esperado
    with st.expander("📖 Ver ejemplo de formato esperado"):
        ejemplo_df = pd.DataFrame({
            'ID': [1, 2, 3],
            'CSAT': [
                'Gracias por su excelente servicio',
                'Esto es terrible, estoy muy enojado',
                'El sistema no funciona'
            ],
            'Fecha': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        st.dataframe(ejemplo_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "💡 Análisis de Sentimientos v1.2 | Desarrollado con Streamlit"
    "</div>",
    unsafe_allow_html=True
)
