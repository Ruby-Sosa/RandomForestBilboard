
import streamlit as st
import pandas as pd
import joblib
import requests

# ==========================================
# 1. CONFIGURACIÓN DE LA APLICACIÓN
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
# 3. CARGAR DATASET ORIGINAL
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

    # Limpiar duraciones igual que en el entrenamiento
    mediana = datos.loc[
        datos["durationSeg"] <= 600,
        "durationSeg"
    ].median()

    datos.loc[
        datos["durationSeg"] > 600,
        "durationSeg"
    ] = mediana

    # Convertir fecha y extraer sus componentes
    fecha = pd.to_datetime(
        datos["chart_date"].astype(str),
        format="%Y%m%d"
    )

    datos["anio_lista"] = fecha.dt.year
    datos["mes_lista"] = fecha.dt.month
    datos["dia_lista"] = fecha.dt.day

    datos.drop(columns=["id", "chart_date"], inplace=True)

    # Aplicar One-Hot Encoding
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
# 4. BUSCAR CANCIÓN EN ITUNES
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
# 5. FORMULARIO DE BÚSQUEDA
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
            st.warning("No se encontró la canción en iTunes.")

        else:
            # Buscar coincidencia por título y artista
            coincidencia = None

            for cancion in resultados:
                nombre = cancion.get("trackName", "").upper()
                interprete = cancion.get("artistName", "").upper()

                if (
                    titulo.strip().upper() in nombre
                    and artista.strip().upper() in interprete
                ):
                    coincidencia = cancion
                    break

            if coincidencia is None:
                st.warning(
                    "Se encontraron canciones similares, "
                    "pero no una coincidencia suficiente."
                )

            else:
                st.success("¡Canción encontrada!")

                # ==========================================
                # 6. MOSTRAR INFORMACIÓN
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

                # Vista previa
                vista_previa = coincidencia.get("previewUrl")

                if vista_previa:
                    st.subheader("Vista previa")
                    st.audio(vista_previa)
                else:
                    st.info("Esta canción no tiene vista previa disponible.")

                # ==========================================
                # 7. BUSCAR CARACTERÍSTICAS EN EL DATASET
                # ==========================================

                titulo_busqueda = coincidencia.get(
                    "trackName", ""
                ).strip().upper()

                artista_busqueda = coincidencia.get(
                    "artistName", ""
                ).strip().upper()

                coincidencias = datos[
                    (datos["title"] == titulo_busqueda)
                    &
                    (datos["artist"] == artista_busqueda)
                ]

                st.subheader("Predicción del modelo")

                if coincidencias.empty:
                    st.warning(
                        "La canción no se encuentra en nuestro "
                        "dataset de entrenamiento. No podemos "
                        "realizar una predicción confiable con "
                        "las características necesarias."
                    )

                else:
                    # Recuperar las características
                    registro = coincidencias.iloc[[0]].copy()

                    # Separar la variable objetivo
                    X_cancion = registro.drop(
                        columns=["top", "title", "artist"]
                    )

                    # Asegurar el mismo orden de columnas
                    X_cancion = X_cancion.reindex(
                        columns=columnas,
                        fill_value=0
                    )

                    # Realizar predicción
                    prediccion = modelo.predict(X_cancion)[0]

                    probabilidades = modelo.predict_proba(X_cancion)[0]

                    probabilidad_top = probabilidades[
                        list(modelo.classes_).index(1)
                    ]

                    if prediccion == 1:
                        st.success(
                            "🎉 El modelo predice que la canción "
                            "llegará al Billboard Top."
                        )
                    else:
                        st.info(
                            "El modelo predice que la canción "
                            "no llegará al Billboard Top."
                        )

                    st.metric(
                        "Probabilidad estimada de la clase Top",
                        f"{probabilidad_top * 100:.2f}%"
                    )

                    st.caption(
                        "Resultado generado por el modelo Random Forest "
                        "con las características del dataset. "
                        "La probabilidad no garantiza un resultado real."
                    )

# ==========================================
# 8. INFORMACIÓN ADICIONAL
# ==========================================

st.divider()

st.caption(
    "Proyecto académico de aprendizaje supervisado. "
    "Modelo: Random Forest Classifier."
)
