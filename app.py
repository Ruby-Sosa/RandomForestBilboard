
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
    layout="centered"
)

st.title("🎵 Billboard Predictor")
st.write(
    "Busca una canción por su título y artista "
    "y descubre si llegó al Billboard Top según nuestro modelo."
)

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
# 4. NORMALIZAR TEXTO PARA COMPARACIONES
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
# 5. BUSCAR CANCIÓN EN ITUNES
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
# 6. FORMULARIO DE BÚSQUEDA
# ==========================================

st.subheader("Buscar canción")

titulo = st.text_input("Nombre de la canción")
artista = st.text_input("Nombre del artista")

if st.button("Buscar y predecir", type="primary"):

    if not titulo.strip() or not artista.strip():

        st.warning("Escribe el título de la canción y el artista.")

    else:

        with st.spinner("Buscando canción..."):

            try:
                resultados = buscar_itunes(titulo, artista)

            except Exception as e:
                st.error(f"Error al consultar iTunes: {e}")
                st.stop()

        if not resultados:

            st.warning("No se encontraron canciones en iTunes.")

        else:

            # ==========================================
            # 7. MOSTRAR RESULTADOS DE ITUNES
            # ==========================================

            st.success(
                f"Se encontraron {len(resultados)} resultados."
            )

            opciones = []

            for i, cancion in enumerate(resultados):

                nombre = cancion.get("trackName", "Sin título")
                interprete = cancion.get("artistName", "Artista desconocido")

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
            # 8. MOSTRAR INFORMACIÓN
            # ==========================================

            st.subheader("Información de la canción")

            st.write(
                "**Título:**",
                coincidencia.get("trackName", "No disponible")
            )

            st.write(
                "**Artista:**",
                coincidencia.get("artistName", "No disponible")
            )

            st.write(
                "**Álbum:**",
                coincidencia.get("collectionName", "No disponible")
            )

            st.write(
                "**Género:**",
                coincidencia.get("primaryGenreName", "No disponible")
            )

            imagen = coincidencia.get("artworkUrl100")

            if imagen:
                st.image(imagen, width=180)

            vista_previa = coincidencia.get("previewUrl")

            if vista_previa:
                st.subheader("Vista previa")
                st.audio(vista_previa)
            else:
                st.info("Esta canción no tiene vista previa disponible.")

            # ==========================================
            # 9. BUSCAR EN EL DATASET
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

            st.subheader("Predicción del modelo")

            if coincidencias.empty:

                st.warning(
                    "La canción fue encontrada en iTunes, "
                    "pero no aparece en nuestro dataset de Billboard. "
                    "Por eso no podemos generar una predicción "
                    "con el modelo entrenado."
                )

            else:

                registro = coincidencias.iloc[[0]].copy()

                # Eliminar columnas que no utiliza el modelo
                registro = registro.drop(
                    columns=[
                        "top",
                        "title",
                        "artist",
                        "titulo_normalizado",
                        "artista_normalizado"
                    ]
                )

                # Alinear las características
                X_cancion = registro.reindex(
                    columns=columnas,
                    fill_value=0
                )

                # Predicción
                prediccion = modelo.predict(X_cancion)[0]

                probabilidades = modelo.predict_proba(X_cancion)[0]

                probabilidad_top = probabilidades[
                    list(modelo.classes_).index(1)
                ]

                if prediccion == 1:

                    st.success(
                        "🎉 El modelo identifica esta canción "
                        "como parte de la clase Billboard Top."
                    )

                else:

                    st.info(
                        "El modelo identifica esta canción "
                        "como parte de la clase que no llegó al Billboard Top."
                    )

                st.metric(
                    "Probabilidad estimada de la clase Top",
                    f"{probabilidad_top * 100:.2f}%"
                )

                st.caption(
                    "Resultado generado por el modelo Random Forest "
                    "utilizando las características del dataset. "
                    "La probabilidad no garantiza un resultado real."
                )

# ==========================================
# 10. INFORMACIÓN ADICIONAL
# ==========================================

st.divider()

st.caption(
    "Proyecto académico de aprendizaje supervisado. "
    "Modelo: Random Forest Classifier."
)
