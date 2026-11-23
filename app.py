
import streamlit as st
import pandas as pd
import joblib
import requests
import unicodedata
import re
from datetime import datetime

# ==========================================
# 1. CONFIGURACIÓN Y DISEÑO
# ==========================================

st.set_page_config(
    page_title="Billboard Predictor",
    page_icon="🎵",
    layout="wide"
)

# Diseño personalizado
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #10101c, #19152b, #10101c);
    color: #ffffff;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #ffffff !important;
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
    margin-bottom: 8px;
}

.hero p {
    font-size: 16px;
    color: #f0e7ff;
}

.section-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(190, 160, 255, 0.20);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 20px;
}

.info-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(190, 160, 255, 0.20);
    border-radius: 14px;
    padding: 16px;
    min-height: 95px;
    margin-bottom: 12px;
}

.info-label {
    color: #bda8ec;
    font-size: 13px;
    margin-bottom: 7px;
}

.info-value {
    color: white;
    font-size: 17px;
    font-weight: 600;
    overflow-wrap: anywhere;
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

.result-positive h2, .result-negative h2 {
    margin-top: 0;
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
    border: none;
}

div[data-baseweb="input"] input {
    background-color: #252238;
    color: white;
}

div[data-baseweb="select"] > div {
    background-color: #252238;
}

hr {
    border-color: rgba(190, 160, 255, 0.25);
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
# 2. CARGAR MODELO Y COLUMNAS
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
# 3. CARGAR DATASET
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

    # One-Hot Encoding
    datos = pd.get_dummies(
        datos,
        columns=["mood", "tempo", "genre", "artist_type"],
        dtype=int
    )

    return datos

try:
    datos = cargar_datos()
except Exception as e:
    st.error(f"No se pudo cargar el dataset: {e}")
    st.stop()

# ==========================================
# 4. NORMALIZAR TEXTO
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
# 5. BUSCAR EN ITUNES
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
# 6. FUNCIONES PARA MOSTRAR INFORMACIÓN
# ==========================================

def mostrar_tarjeta(etiqueta, valor):

    if valor is None or valor == "":
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
# 7. FORMULARIO DE BÚSQUEDA
# ==========================================

st.markdown("""
<div class="section-card">
    <h3>🔎 Buscar una canción</h3>
    <p>Ingresa el título y el nombre del artista para comenzar.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    titulo = st.text_input(
        "Nombre de la canción",
        placeholder="Ej. Shape of You"
    )

with col2:
    artista = st.text_input(
        "Nombre del artista",
        placeholder="Ej. Ed Sheeran"
    )

buscar = st.button(
    "🎵 Buscar canción y generar pronóstico",
    type="primary"
)

# ==========================================
# 8. PROCESAR BÚSQUEDA
# ==========================================

if buscar:

    if not titulo.strip() or not artista.strip():

        st.warning("Escribe el título de la canción y el artista.")

    else:

        with st.spinner("Buscando información de la canción..."):

            try:
                resultados = buscar_itunes(titulo, artista)

            except Exception as e:
                st.error(f"Error al consultar iTunes: {e}")
                st.stop()

        if not resultados:

            st.warning("No se encontraron canciones en iTunes.")

        else:

            st.success(
                f"Se encontraron {len(resultados)} resultados."
            )

            opciones = []

            for cancion in resultados:

                nombre = cancion.get("trackName", "Sin título")
                interprete = cancion.get(
                    "artistName",
                    "Artista desconocido"
                )

                opciones.append(
                    f"{nombre} - {interprete}"
                )

            seleccion = st.selectbox(
                "Selecciona la canción correcta:",
                range(len(opciones)),
                format_func=lambda i: opciones[i]
            )

            coincidencia = resultados[seleccion]

            # ==========================================
            # 9. FICHA TÉCNICA
            # ==========================================

            st.divider()

            st.markdown(
                "## 🎧 Ficha técnica",
                help="Datos recuperados de iTunes y del conjunto de datos Billboard."
            )

            st.caption(
                "Datos recuperados de la búsqueda musical y, "
                "cuando existe coincidencia, del dataset utilizado."
            )

            col_imagen, col_datos = st.columns([1, 2])

            with col_imagen:

                imagen = coincidencia.get("artworkUrl100")

                if imagen:
                    st.image(imagen, use_container_width=True)
                else:
                    st.info("Portada no disponible.")

            with col_datos:

                st.markdown(
                    f"### {coincidencia.get('trackName', 'Sin título')}"
                )

                st.markdown(
                    f"**{coincidencia.get('artistName', 'Artista desconocido')}**"
                )

                st.write(
                    "Álbum:",
                    coincidencia.get("collectionName", "No disponible")
                )

                st.write(
                    "Género:",
                    coincidencia.get("primaryGenreName", "No disponible")
                )

                st.write(
                    "Fecha de lanzamiento:",
                    coincidencia.get("releaseDate", "No disponible")[:10]
                )

                st.write(
                    "Duración:",
                    convertir_duracion(
                        coincidencia.get("trackTimeMillis")
                    )
                )

            # Vista previa
            vista_previa = coincidencia.get("previewUrl")

            if vista_previa:
                st.markdown("#### ▶️ Escuchar vista previa")
                st.audio(vista_previa)

            # ==========================================
            # 10. BUSCAR EN EL DATASET
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
                (datos_busqueda["titulo_normalizado"] == titulo_busqueda)
                &
                (datos_busqueda["artista_normalizado"] == artista_busqueda)
            ]

            # ==========================================
            # 11. DATOS DEL DATASET
            # ==========================================

            st.markdown("### 📊 Datos recuperados de Billboard")

            if coincidencias.empty:

                st.info(
                    "Esta canción no aparece en el dataset de entrenamiento. "
                    "Se muestran los datos recuperados de iTunes, "
                    "pero no es posible generar un pronóstico con el modelo."
                )

                registro = None

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

                # Mostrar características categóricas originales
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
                        "Duración del dataset",
                        convertir_duracion(
                            registro.iloc[0]["durationSeg"] * 1000
                        )
                    )

                with col2:
                    mostrar_tarjeta(
                        "Género del dataset",
                        registro.iloc[0]["genre"]
                    )

            # ==========================================
            # 12. RESULTADO DEL PRONÓSTICO
            # ==========================================

            st.divider()

            st.markdown("## 🔮 Resultado del pronóstico")

            if registro is None:

                st.warning(
                    "No se puede generar un pronóstico porque "
                    "no se encontraron las características de esta "
                    "canción en el dataset utilizado para entrenar "
                    "el modelo Random Forest."
                )

            else:

                # Preparar características
                X_cancion = registro.drop(
                    columns=[
                        "top",
                        "title",
                        "artist",
                        "titulo_normalizado",
                        "artista_normalizado"
                    ]
                )

                X_cancion = X_cancion.reindex(
                    columns=columnas,
                    fill_value=0
                )

                # Predicción
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
                            "El modelo predice con una probabilidad "
                            "alta que esta canción pertenece a la "
                            "clase Billboard Top."
                        )

                    elif confianza >= 0.60:
                        mensaje = "¡Hay buenas posibilidades! ✨"
                        descripcion = (
                            "El modelo estima que esta canción "
                            "podría pertenecer al Billboard Top."
                        )

                    else:
                        mensaje = "¡El pronóstico es favorable! 🎵"
                        descripcion = (
                            "El modelo se inclina por la clase "
                            "Billboard Top, aunque su confianza "
                            "es limitada."
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

                # Resultado negativo
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

                # Indicador de confianza de la clase predicha
                st.markdown("#### Nivel de confianza de la clase predicha")

                st.progress(float(confianza))

                st.write(
                    f"Confianza estimada: {confianza * 100:.2f}%"
                )

                st.caption(
                    "El porcentaje es una estimación del modelo "
                    "Random Forest y no garantiza que la canción "
                    "alcance una posición específica en Billboard. "
                    "El resultado depende de los datos utilizados "
                    "durante el entrenamiento."
                )

# ==========================================
# 13. PIE DE PÁGINA
# ==========================================

st.divider()

st.markdown("""
<div style="text-align:center; padding:15px;">
    <p style="color:#bda8ec;">
        🎵 Billboard Predictor
    </p>
    <p style="font-size:13px; color:#aaa;">
        Proyecto académico de aprendizaje supervisado
        <br>
        Modelo de clasificación: Random Forest
    </p>
</div>
""", unsafe_allow_html=True)
