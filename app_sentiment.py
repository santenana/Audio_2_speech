import streamlit as st
import pandas as pd
import io
from langdetect import detect
import nltk
import re
import unicodedata
import os

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
Esta aplicación analiza sentimientos en textos y genera 11 nuevas columnas con métricas de análisis.
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
    
    # Detectar idioma para cada fila
    progress_bar = st.progress(0, text="Detectando idiomas...")
    
    df_result['_lang_temp'] = df_result[text_column].apply(
        lambda x: detect_language_safe(str(x)) if pd.notna(x) else 'es'
    )
    progress_bar.progress(10, text="Idiomas detectados")
    
    # Inicializar las 11 columnas nuevas
    progress_bar.progress(15, text="Inicializando columnas...")
    df_result['polite_words'] = 0
    df_result['rude_words'] = 0
    df_result['technical_terms'] = 0
    df_result['toxic_words'] = 0
    df_result['desagrado'] = 0
    df_result['frustracion'] = 0
    df_result['gratitud'] = 0
    df_result['indiferencia'] = 0
    df_result['satisfaccion'] = 0
    df_result['rabia_ira'] = 0
    df_result['amenazas'] = 0
    
    # Procesar cada fila
    total_rows = len(df_result)
    
    for idx, row in df_result.iterrows():
        # Actualizar barra de progreso
        if idx % max(1, total_rows // 20) == 0:
            progress = 15 + int((idx / total_rows) * 80)
            progress_bar.progress(progress, text=f"Procesando fila {idx + 1}/{total_rows}...")
        
        text = row[text_column]
        if pd.isna(text) or not isinstance(text, str):
            continue
        
        lang = row['_lang_temp']
        
        # Analizar cada categoría
        df_result.at[idx, 'polite_words'] = count_words_in_text(text, word_lists['polite_words'], lang)
        df_result.at[idx, 'rude_words'] = count_words_in_text(text, word_lists['rude_words'], lang)
        df_result.at[idx, 'technical_terms'] = count_words_in_text(text, word_lists['technical_terms'], lang)
        df_result.at[idx, 'toxic_words'] = count_words_in_text(text, word_lists['toxic_words'], lang)
        df_result.at[idx, 'desagrado'] = count_words_in_text(text, word_lists['desagrado'], lang)
        df_result.at[idx, 'frustracion'] = count_words_in_text(text, word_lists['frustracion'], lang)
        df_result.at[idx, 'gratitud'] = count_words_in_text(text, word_lists['gratitud'], lang)
        df_result.at[idx, 'indiferencia'] = count_words_in_text(text, word_lists['indiferencia'], lang)
        df_result.at[idx, 'satisfaccion'] = count_words_in_text(text, word_lists['satisfaccion'], lang)
        df_result.at[idx, 'rabia_ira'] = count_words_in_text(text, word_lists['rabia_ira'], lang)
        df_result.at[idx, 'amenazas'] = count_words_in_text(text, word_lists['amenazas'], lang)
    
    # Eliminar columna temporal de idioma
    df_result = df_result.drop('_lang_temp', axis=1)
    
    progress_bar.progress(100, text="✅ Procesamiento completado!")
    
    return df_result

# ============================================================================
# INTERFAZ DE STREAMLIT
# ============================================================================

# Cargar diccionarios
word_lists = load_word_lists()

# Sidebar con información
# with st.sidebar:
#     st.header("ℹ️ Información")
#     st.markdown("""
#     ### Columnas generadas:
    
#     1. **polite_words**: Palabras educadas
#     2. **rude_words**: Palabras groseras
#     3. **technical_terms**: Términos técnicos
#     4. **toxic_words**: Palabras tóxicas
#     5. **desagrado**: Expresiones de desagrado
#     6. **frustracion**: Expresiones de frustración
#     7. **gratitud**: Expresiones de gratitud
#     8. **indiferencia**: Expresiones de indiferencia
#     9. **satisfaccion**: Expresiones de satisfacción
#     10. **rabia_ira**: Expresiones de rabia/ira
#     11. **amenazas**: Expresiones de amenazas
#     """)
    
#     st.markdown("---")
#     st.markdown("### 📝 Formatos soportados")
#     st.markdown("- CSV (.csv)")
#     st.markdown("- Excel (.xlsx, .xls)")

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
                
                # Guardar en session_state
                st.session_state['df_resultado'] = df_resultado
                st.session_state['selected_column'] = selected_column
                
                st.success("✅ ¡Análisis completado exitosamente!")
                # st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error durante el procesamiento: {str(e)}")
                st.exception(e)
        
        # Mostrar resultados si existen
        if 'df_resultado' in st.session_state:
            st.header("4️⃣ Resultados")
            
            df_resultado = st.session_state['df_resultado']
            
            # Crear pestañas para organizar los resultados
            tab1, tab3, tab4 = st.tabs(["📈 Estadísticas", "📊 Distribución", "📥 Descargar"])
            
            # PESTAÑA 1: Estadísticas Generales
            with tab1:
                st.subheader("📈 Estadísticas Generales")
                
                cols = st.columns(4)
                
                total_polite = int(df_resultado['polite_words'].sum())
                total_rude = int(df_resultado['rude_words'].sum())
                total_gratitud = int(df_resultado['gratitud'].sum())
                total_rabia = int(df_resultado['rabia_ira'].sum())
                
                cols[0].metric("💚 Palabras Educadas", total_polite)
                cols[1].metric("😡 Palabras Groseras", total_rude)
                cols[2].metric("🙏 Expresiones de Gratitud", total_gratitud)
                cols[3].metric("😤 Expresiones de Rabia", total_rabia)
                
                st.markdown("---")
                
                # Métricas adicionales
                cols2 = st.columns(4)
                total_tech = int(df_resultado['technical_terms'].sum())
                total_toxic = int(df_resultado['toxic_words'].sum())
                total_satisfaccion = int(df_resultado['satisfaccion'].sum())
                total_amenazas = int(df_resultado['amenazas'].sum())
                
                cols2[0].metric("🔧 Términos Técnicos", total_tech)
                cols2[1].metric("☠️ Palabras Tóxicas", total_toxic)
                cols2[2].metric("😊 Satisfacción", total_satisfaccion)
                cols2[3].metric("⚠️ Amenazas", total_amenazas)
            
            # PESTAÑA 2: Vista de Resultados
            # with tab2:
            #     st.subheader("📋 Vista de Resultados")
            #     st.dataframe(df_resultado, use_container_width=True, height=400)
            
            # PESTAÑA 3: Distribución de Sentimientos
            with tab3:
                st.subheader("📊 Distribución de Sentimientos")
                
                sentiment_cols = ['polite_words', 'rude_words', 'technical_terms', 'toxic_words', 
                                'desagrado', 'frustracion', 'gratitud', 'indiferencia', 
                                'satisfaccion', 'rabia_ira', 'amenazas']
                
                sentiment_totals = {col: int(df_resultado[col].sum()) for col in sentiment_cols}
                sentiment_df = pd.DataFrame({
                    'Categoría': list(sentiment_totals.keys()),
                    'Total': list(sentiment_totals.values())
                }).sort_values('Total', ascending=False)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.bar_chart(sentiment_df.set_index('Categoría'))
                
                # with col2:
                #     st.dataframe(sentiment_df, use_container_width=True, hide_index=True)
            
            # PESTAÑA 4: Descargar Resultados
            
            
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
    "💡 Análisis de Sentimientos v1.1 | Desarrollado con Streamlit"
    "</div>",
    unsafe_allow_html=True
)





