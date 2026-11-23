

                mostrar_tarjeta("Género", registro.iloc[0]["genre"])

        # ==========================================
        # 13. RESULTADO DEL PRONÓSTICO
        # ==========================================

        st.divider()
        st.markdown("## 🔮 Resultado del pronóstico")

        if registro is None:
            st.markdown("""
            <div class="aviso-pronostico">
                <h3>🔮 Pronóstico no disponible</h3>
                <p>
                    Nuestro modelo Random Forest necesita características
                    del dataset de entrenamiento que no están disponibles
                    para esta canción.
                </p>
                <p>
                    <strong>¡No te preocupes!</strong>
                    Puedes buscar otra canción que se encuentre registrada
                    en el dataset de Billboard.
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
            )

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
                probabilidad_top if prediccion == 1
                else 1 - probabilidad_top
            )

            if prediccion == 1:
                if confianza >= 0.80:
                    mensaje = "¡Casi felicidades! 🎉"
                    descripcion = (
                        "El modelo predice con una probabilidad alta que "
                        "esta canción pertenece a la clase Billboard Top."
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
                    <p>Probabilidad estimada de pertenecer a la clase Billboard Top</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()

            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2>El pronóstico no es favorable 🎧</h2>
                    <p>
                        El modelo predice que esta canción no pertenece
                        a la clase Billboard Top.
                    </p>
                    <div class="result-number">
                        {probabilidad_top * 100:.2f}%
                    </div>
                    <p>Probabilidad estimada de pertenecer a la clase Billboard Top</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Nivel de confianza de la clase predicha")
            st.progress(float(confianza))
            st.write(f"Confianza estimada: {confianza * 100:.2f}%")
            st.caption(
                "El porcentaje es una estimación del modelo Random Forest "
                "y no garantiza una posición específica en Billboard."
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


