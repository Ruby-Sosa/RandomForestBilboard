
import streamlit as st
import pandas as pd
import joblib
import requests
import unicodedata
import re

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

st.set_page_config(
    page_title="Billboard Predictor",
    page_icon="🎵",
    layout="wide"
)

# ==========================================
# 2. DISEÑO PERSONALIZADO
# ==========================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10101c, #19152b, #10101c);
    color: white;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    color: white !important;
}

p, label, span {
    color: #e3e0ee;
}

.hero {
    background: linear-gradient(120deg, #5425a8, #9b42d5);
    padding: 35px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.25);
}

.hero h1 {
    font-size: 38px;
}

.hero p {
    color: #f0e7ff;
}

.mensaje-exito {
    background: linear-gradient(120deg, #30234f, #54347a);
    border: 1px solid #a77bea;
    border-radius: 12px;
    padding: 13px 18px;
    color: #ffffff !important;
    font-weight: 600;
    margin: 12px 0 18px 0;
}

.info-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(190,160,255,0.20);
    border-radius: 14px;
    padding: 16px;
    min-height: 90px;
    margin-bottom: 12px;
}

.info-label {
    color: #c5a8f5 !important;
    font-size: 13px;
    margin-bottom: 7px;
}

.info-value {
    color: #ffffff !important;
    font-size: 17px;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.aviso-personalizado {
    background: linear-gradient(120deg, #30234f, #432c68);
    border: 1px solid #a77bea;
    border-left: 5px solid #c49aff;
    border-radius: 13px;
    padding: 18px 20px;
    margin: 15px 0;
    color: #ffffff !important;
    font-size: 15px;
    line-height: 1.7;
}

.aviso-personalizado strong {
    color: #e2caff !important;
}

.aviso-pronostico {
    background: linear-gradient(120deg, #38264f, #512f68);
    border: 1px solid #c084fc;
    border-radius: 18px;
    padding: 25px;
    margin: 15px 0;
    color: #ffffff !important;
    font-size: 16px;
    line-height: 1.7;
}

.aviso-pronostico h3 {
    color: #e9d5ff !important;
}

.aviso-pronostico p {
    color: #ffffff !important;
}

.result-positive {
    background: linear-gradient(135deg, #173f35, #17634b);
    border: 1px solid #3fbd8b;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin-top: 15px;
}

.result-negative {
    background: linear-gradient(135deg, #3c253d, #512d50);
    border: 1px solid #b77bc8;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin-top: 15px;
}

.result-positive h2,
.result-negative h2 {
    color: white !important;
}

.result-number {
    font-size: 38px;
    font-weight: bold;
    color: white;
    margin: 12px 0;
}

div.stButton > button {
    background: linear-gradient(90deg, #7839cf, #a34ee0);
    color: white;
    border: none;
    border-radius: 12px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #9254e8, #bb68ed);
    color: white;
}

div[data-baseweb="input"] input {
    background-color: #252238;
    color: white;
}

div[data-baseweb="select"] > div {
    background-color: #252238;
}

hr {
    border-color: rgba(190,160,255,0.25);
}
</style>
""", unsafe_allow_html=True)

# Encabezado

st.markdown("""
<div class="hero">
    <h1>🎵 Billboard Predictor</h1>
    <p>
        Descubre qué tan probable es que una canción
        pertenezca al Billboard Top mediante Inteligencia Artificial.
    </p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. CARGAR MODELO Y COLUMNAS
# ==========================================

@st.cache_resource
def cargar_modelo():

    modelo = joblib.load("modelo_random_forest.joblib")
    columnas = joblib.load("columnas_modelo.joblib")

    return modelo, columnas

try:
    modelo, columnas = cargar_modelo()

except Exception as e:
    st.error(f"No se pudieron cargar los archivos del modelo: {e}")
    st.stop()

# ==========================================
# 4. CARGAR DATASET
# ==========================================

@st.cache_data
def cargar_datos():

    url = (
        "https://raw.githubusercontent.com/"
        "gerrymay/fcaData/refs/heads/master/"
        "Billboard/artists_billboard.csv"
    )

    datos = pd.read_csv(url)

    # Limpiar textos

    columnas_texto = datos.select_dtypes(include="object").columns

    for columna in columnas_texto:
        datos[columna] = datos[columna].str.strip().str.upper()

    # Limpiar duraciones

    mediana = datos.loc[
        datos["durationSeg"] <= 600,
        "durationSeg"
    ].median()

    datos.loc[
        datos["durationSeg"] > 600,
        "durationSeg"
    ] = mediana

    # Procesar fechas

    fecha = pd.to_datetime(
        datos["chart_date"].astype(str),
        format="%Y%m%d"
    )

    datos["anio_lista"] = fecha.dt.year
    datos["mes_lista"] = fecha.dt.month
    datos["dia_lista"] = fecha.dt.day

    datos.drop(
        columns=["id", "chart_date"],
        inplace=True
    )

    # Conservamos las categorías originales para mostrarlas.
    # El One-Hot Encoding se aplica antes de la predicción.

    return datos

try:
    datos = cargar_datos()

except Exception as e:
    st.error(f"No se pudo cargar el dataset: {e}")
    st.stop()

# ==========================================
# 5. NORMALIZAR TEXTO
# ==========================================

def normalizar_texto(texto):

    texto = str(texto).upper().strip()

    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caracter for caracter in texto
        if unicodedata.category(caracter) != "Mn"
    )

    texto = re.sub(r"[^A-Z0-9 ]", " ", texto)

    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()

# ==========================================
# 6. BUSCAR EN ITUNES
# ==========================================

def buscar_itunes(titulo, artista):

    url = "https://itunes.apple.com/search"

    parametros = {
        "term": f"{titulo} {artista}",
        "entity": "song",
        "limit": 10,
        "country": "US"
    }

    respuesta = requests.get(
        url,
        params=parametros,
        timeout=15
    )

    respuesta.raise_for_status()

    return respuesta.json().get("results", [])

# ==========================================
# 7. FUNCIONES AUXILIARES
# ==========================================

def mostrar_tarjeta(etiqueta, valor):

    if valor is None or pd.isna(valor) or valor == "":
        valor = "No disponible"

    st.markdown(f"""
    <div class="info-card">
        <div class="info-label">{etiqueta}</div>
        <div class="info-value">{valor}</div>
    </div>
    """, unsafe_allow_html=True)


def convertir_duracion(milisegundos):

    if not milisegundos:
        return "No disponible"

    segundos = int(milisegundos / 1000)

    minutos = segundos // 60

    segundos_restantes = segundos % 60

    return f"{minutos}:{segundos_restantes:02d} min"

# ==========================================
# 8. FORMULARIO DE BÚSQUEDA
# ==========================================

st.markdown("## 🔎 Buscar una canción")

st.write(
    "Ingresa el título y el nombre del artista "
    "para consultar su información."
)

col1, col2 = st.columns(2)

with col1:
    titulo = st.text_input(
        "Nombre de la canción",
        placeholder="Ej. Shape of You",
        key="titulo_cancion"
    )

with col2:
    artista = st.text_input(
        "Nombre del artista",
        placeholder="Ej. Ed Sheeran",
        key="artista_cancion"
    )

buscar = st.button(
    "🎵 Buscar canción y generar pronóstico",
    type="primary"
)

# ==========================================
# 9. PROCESAR BÚSQUEDA
# ==========================================

# Guardar los resultados para que no desaparezcan
# al cambiar la canción seleccionada.

if "resultados_itunes" not in st.session_state:
    st.session_state.resultados_itunes = []

if "busqueda_realizada" not in st.session_state:
    st.session_state.busqueda_realizada = False

if buscar:

    if not titulo.strip() or not artista.strip():

        st.warning("Escribe el título de la canción y el artista.")

        st.session_state.resultados_itunes = []
        st.session_state.busqueda_realizada = False

    else:

        with st.spinner("Buscando información de la canción..."):

            try:

                resultados = buscar_itunes(titulo, artista)

                st.session_state.resultados_itunes = resultados
                st.session_state.busqueda_realizada = True

                # Reiniciar la selección al realizar una nueva búsqueda
                st.session_state.cancion_seleccionada = 0

            except Exception as e:

                st.error(f"Error al consultar iTunes: {e}")

                st.session_state.resultados_itunes = []
                st.session_state.busqueda_realizada = False

# Mostrar los resultados guardados incluso después de cambiar la selección

if st.session_state.busqueda_realizada:

    resultados = st.session_state.resultados_itunes

    if not resultados:

        st.warning("No se encontraron canciones en iTunes.")

    else:

        st.markdown(
            f'<div class="mensaje-exito">🎵 Se encontraron '
            f'{len(resultados)} resultados.</div>',
            unsafe_allow_html=True
        )

        opciones = []

        for cancion in resultados:

            nombre = cancion.get("trackName", "Sin título")

            interprete = cancion.get(
                "artistName",
                "Artista desconocido"
            )

            opciones.append(f"{nombre} - {interprete}")

        seleccion = st.selectbox(
            "Selecciona la canción correcta:",
            range(len(opciones)),
            format_func=lambda i: opciones[i],
            key="cancion_seleccionada"
        )

        coincidencia = resultados[seleccion]

        # ==========================================
        # 10. INFORMACIÓN RECUPERADA DE ITUNES
        # ==========================================

        st.divider()

        st.markdown("## 🎧 Información recuperada de iTunes")

        st.caption("Datos proporcionados por el catálogo de iTunes")

        col_imagen, col_datos = st.columns([1, 2])

        with col_imagen:

            imagen = coincidencia.get("artworkUrl100")

            if imagen:

                st.image(
                    imagen,
                    use_container_width=True
                )

            else:

                st.info("Portada no disponible.")

        with col_datos:

            st.markdown(
                f"### {coincidencia.get('trackName', 'Sin título')}"
            )

            st.markdown(
                f"**{coincidencia.get('artistName', 'Artista desconocido')}**"
            )

            # Solo mostramos la fecha en esta sección
            mostrar_tarjeta(
                "Fecha de lanzamiento",
                coincidencia.get("releaseDate", "")[:10]
            )

        # Vista previa de audio

        vista_previa = coincidencia.get("previewUrl")

        if vista_previa:

            st.markdown("#### ▶️ Escuchar vista previa")

            st.audio(vista_previa)

        else:

            st.info("Esta canción no tiene vista previa disponible.")

        # ==========================================
        # 11. BUSCAR EN DATASET BILLBOARD
        # ==========================================

        titulo_busqueda = normalizar_texto(
            coincidencia.get("trackName", "")
        )

        artista_busqueda = normalizar_texto(
            coincidencia.get("artistName", "")
        )

        datos_busqueda = datos.copy()

        datos_busqueda["titulo_normalizado"] = (
            datos_busqueda["title"].apply(normalizar_texto)
        )

        datos_busqueda["artista_normalizado"] = (
            datos_busqueda["artist"].apply(normalizar_texto)
        )

        coincidencias = datos_busqueda[
            (
                datos_busqueda["titulo_normalizado"]
                == titulo_busqueda
            )
            &
            (
                datos_busqueda["artista_normalizado"]
                == artista_busqueda
            )
        ]

        # ==========================================
        # 12. DATOS RECUPERADOS DE BILLBOARD
        # ==========================================

        st.divider()

        st.markdown("## 📊 Datos recuperados de Billboard")

        if coincidencias.empty:

            registro = None

            st.markdown("""
            <div class="aviso-personalizado">
                <strong>🎵 La canción no está registrada en Billboard</strong><br>
                No encontramos esta canción en el dataset utilizado para
                entrenar el modelo. A continuación se muestran los datos
                que sí fue posible recuperar de iTunes.
            </div>
            """, unsafe_allow_html=True)

            # Mostrar los datos de iTunes en esta sección

            fecha_lanzamiento = coincidencia.get("releaseDate", "")

            if fecha_lanzamiento:
                anio_lanzamiento = fecha_lanzamiento[:4]
            else:
                anio_lanzamiento = "No disponible"

            col1, col2, col3 = st.columns(3)

            with col1:
                mostrar_tarjeta(
                    "Título",
                    coincidencia.get("trackName")
                )

            with col2:
                mostrar_tarjeta(
                    "Artista",
                    coincidencia.get("artistName")
                )

            with col3:
                mostrar_tarjeta(
                    "Álbum",
                    coincidencia.get("collectionName")
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                mostrar_tarjeta(
                    "Género musical",
                    coincidencia.get("primaryGenreName")
                )

            with col2:
                mostrar_tarjeta(
                    "Año de lanzamiento",
                    anio_lanzamiento
                )

            with col3:
                mostrar_tarjeta(
                    "Duración",
                    convertir_duracion(
                        coincidencia.get("trackTimeMillis")
                    )
                )

            st.caption(
                "Estos datos proceden de iTunes y no sustituyen "
                "las características del dataset de entrenamiento."
            )

        else:

            registro = coincidencias.iloc[[0]].copy()

            col1, col2, col3 = st.columns(3)

            with col1:
                mostrar_tarjeta(
                    "Año de la lista",
                    registro.iloc[0]["anio_lista"]
                )

            with col2:
                mostrar_tarjeta(
                    "Mes de la lista",
                    registro.iloc[0]["mes_lista"]
                )

            with col3:
                mostrar_tarjeta(
                    "Día de la lista",
                    registro.iloc[0]["dia_lista"]
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                mostrar_tarjeta(
                    "Estado de ánimo",
                    registro.iloc[0]["mood"]
                )

            with col2:
                mostrar_tarjeta(
                    "Ritmo",
                    registro.iloc[0]["tempo"]
                )

            with col3:
                mostrar_tarjeta(
                    "Tipo de artista",
                    registro.iloc[0]["artist_type"]
                )

            col1, col2 = st.columns(2)

            with col1:
                mostrar_tarjeta(
                    "Duración",
                    f"{registro.iloc[0]['durationSeg']:.0f} segundos"
                )

            with col2:
                mostrar_tarjeta(
                    "Género",
                    registro.iloc[0]["genre"]
                )

        # ==========================================
        # 13. RESULTADO DEL PRONÓSTICO
        # ==========================================

        st.divider()

        st.markdown("## 🔮 Resultado del pronóstico")

        if registro is None:

            st.markdown("""
            <div class="aviso-pronostico">
                <h3>🔮 No se puede calcular un pronóstico</h3>
                <p>
                    Esta canción no aparece en el dataset de Billboard
                    con el que se entrenó nuestro modelo Random Forest.
                    Por eso no contamos con las características necesarias
                    para calcular una predicción confiable.
                </p>
                <p>
                    <strong>¿Qué puedes hacer?</strong>
                    Prueba con otra canción que sí esté registrada en el
                    dataset. La información de iTunes que aparece arriba
                    sí se recuperó correctamente.
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:

            # Preparar características para el modelo

            X_cancion = registro.drop(
                columns=[
                    "top",
                    "title",
                    "artist",
                    "titulo_normalizado",
                    "artista_normalizado"
                ],
                errors="ignore"
            ).copy()

            # Convertir las categorías al mismo formato usado al entrenar.

            X_cancion = pd.get_dummies(
                X_cancion,
                columns=["mood", "tempo", "genre", "artist_type"],
                dtype=int
            )

            # Mantener exactamente las columnas utilizadas por el modelo.

            X_cancion = X_cancion.reindex(
                columns=columnas,
                fill_value=0
            )

            # Generar predicción

            prediccion = modelo.predict(X_cancion)[0]

            probabilidades = modelo.predict_proba(X_cancion)[0]

            probabilidad_top = probabilidades[
                list(modelo.classes_).index(1)
            ]

            confianza = (
                probabilidad_top
                if prediccion == 1
                else 1 - probabilidad_top
            )

            # Resultado positivo

            if prediccion == 1:

                if confianza >= 0.80:

                    mensaje = "¡Casi felicidades! 🎉"

                    descripcion = (
                        "El modelo predice con una probabilidad alta "
                        "que esta canción pertenece a la clase Billboard Top."
                    )

                elif confianza >= 0.60:

                    mensaje = "¡Hay buenas posibilidades! ✨"

                    descripcion = (
                        "El modelo estima que esta canción podría "
                        "pertenecer al Billboard Top."
                    )

                else:

                    mensaje = "¡El pronóstico es favorable! 🎵"

                    descripcion = (
                        "El modelo se inclina por la clase Billboard Top, "
                        "aunque su confianza es limitada."
                    )

                st.markdown(f"""
                <div class="result-positive">
                    <h2>{mensaje}</h2>
                    <p>{descripcion}</p>
                    <div class="result-number">
                        {probabilidad_top * 100:.2f}%
                    </div>
                    <p>
                        Probabilidad estimada de pertenecer
                        a la clase Billboard Top
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.balloons()

            else:

                st.markdown(f"""
                <div class="result-negative">
                    <h2>El pronóstico no es favorable 🎧</h2>
                    <p>
                        El modelo predice que esta canción
                        no pertenece a la clase Billboard Top.
                    </p>
                    <div class="result-number">
                        {probabilidad_top * 100:.2f}%
                    </div>
                    <p>
                        Probabilidad estimada de pertenecer
                        a la clase Billboard Top
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Barra de confianza

            st.markdown(
                "#### Nivel de confianza de la clase predicha"
            )

            st.progress(float(confianza))

            st.write(
                f"Confianza estimada: {confianza * 100:.2f}%"
            )

            st.caption(
                "El porcentaje es una estimación del modelo "
                "Random Forest y no garantiza una posición "
                "específica en Billboard."
            )

# ==========================================
# 14. PIE DE PÁGINA
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px;">
    <p style="color:#c5a8f5;">
        🎵 Billboard Predictor
    </p>
    <p style="font-size:13px; color:#aaa;">
        Proyecto académico de aprendizaje supervisado
        <br>
        Modelo de clasificación: Random Forest
    </p>
</div>
""", unsafe_allow_html=True)

