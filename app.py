"""
app.py — SolarMotion Tracker & Modeler
Aplicación Streamlit unificada para modelar movimiento solar e intensidad lumínica.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Módulos propios
from ml_engine import AVAILABLE_MODELS, fit_model, compute_derivative
from pdf_generator import generate_pdf
from db_manager import init_db, save_run, get_history, delete_run, clear_history

# ─── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="SolarMotion Tracker & Modeler",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Estilos CSS personalizados (inspirados en stitch/code.html) ─────────────
st.markdown("""
<style>
    /* Tipografía */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

    /* Ocultar header y footer por defecto de Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Navbar personalizada */
    .solar-nav {
        background: white;
        border-bottom: 1px solid #e2e8f0;
        padding: 0.75rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: -1rem -1rem 1.5rem -1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .solar-nav h1 {
        font-family: 'Inter', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        margin: 0;
        color: #0f172a;
    }
    .solar-nav h1 span {
        color: #4f46e5;
    }

    /* Tarjetas de métricas */
    .metric-card {
        background: white;
        border-radius: 1rem;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-label {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Fira Code', monospace;
    }
    .metric-r2 { color: #10b981; }
    .metric-mse { color: #f43f5e; }
    .metric-rmse { color: #f59e0b; }

    /* Tarjeta ecuación */
    .equation-card {
        background: linear-gradient(135deg, #4f46e5, #6366f1);
        color: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(79,70,229,0.25);
    }
    .equation-card .eq-label {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #c7d2fe;
        margin-bottom: 0.5rem;
    }
    .equation-card .eq-text {
        font-family: 'Fira Code', monospace;
        font-size: 1.15rem;
        font-weight: 500;
        overflow-x: auto;
        white-space: nowrap;
    }

    /* Sección del panel izquierdo */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }

    /* Modo Toggle */
    div[data-testid="stRadio"] > label {
        font-weight: 600 !important;
    }

    /* Botón principal */
    .stButton > button[kind="primary"] {
        background: #4f46e5 !important;
        border: none !important;
        border-radius: 1rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(79,70,229,0.2) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #4338ca !important;
    }

    /* Historial items */
    .history-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s;
    }
    .history-item:hover {
        border-color: #a5b4fc;
    }
    .history-badge-luz {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        background: #fef3c7;
        color: #b45309;
        font-size: 0.6rem;
        font-weight: 700;
        border-radius: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .history-badge-sombra {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        background: #e0e7ff;
        color: #4338ca;
        font-size: 0.6rem;
        font-weight: 700;
        border-radius: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Forzar fondo blanco en contenedor principal de Streamlit */
    .stApp, .stMainBlockContainer, section[data-testid="stSidebar"],
    div[data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
    }
    .block-container {
        background-color: #f8fafc !important;
        padding-top: 1rem !important;
    }
    /* Forzar texto oscuro */
    .stMarkdown, .stText, p, span, label, div {
        color: #0f172a;
    }
    /* Forzar fondo blanco en data editor */
    div[data-testid="stDataEditor"] {
        background: white !important;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
    }
    /* Pestañas */
    div[data-testid="stTabs"] button {
        color: #475569 !important;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #4f46e5 !important;
        border-bottom-color: #4f46e5 !important;
    }
    /* Selectbox y sliders */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stSlider"] {
        background: white !important;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Navbar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="solar-nav">
    <div style="display:flex;align-items:center;gap:0.6rem;">
        <div style="width:38px;height:38px;background:#fef3c7;border-radius:0.75rem;
                    display:flex;align-items:center;justify-content:center;font-size:1.3rem;">
            ☀️
        </div>
        <h1>Solar<span>Motion</span></h1>
    </div>
    <div style="font-size:0.8rem;color:#94a3b8;">Tracker & Modeler v1.0</div>
</div>
""", unsafe_allow_html=True)

# ─── Inicializar estado de sesión ────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "evidence_files" not in st.session_state:
    st.session_state.evidence_files = []

# ─── Pestañas principales ────────────────────────────────────────────────────
tab_main, tab_history = st.tabs(["📊 Análisis", "📂 Historial"])

# =============================================================================
# TAB 1 — ANÁLISIS PRINCIPAL
# =============================================================================
with tab_main:
    col_left, col_right = st.columns([1, 2], gap="large")

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL IZQUIERDO: Controles
    # ─────────────────────────────────────────────────────────────────────────
    with col_left:
        # -- Modo de Experimento --
        st.markdown('<p class="section-label">Modo de Experimento</p>', unsafe_allow_html=True)
        mode = st.radio(
            "Selecciona el modo",
            ["☀️ Modo Luz (Luxómetro)", "📏 Modo Sombra (Gnomon)"],
            horizontal=True,
            label_visibility="collapsed",
        )
        is_shadow = "Sombra" in mode

        st.divider()

        # -- Selector de Modelo --
        st.markdown('<p class="section-label">Algoritmo de ML</p>', unsafe_allow_html=True)
        model_name = st.selectbox(
            "Modelo", AVAILABLE_MODELS, label_visibility="collapsed"
        )

        # -- Hiperparámetros dinámicos --
        st.markdown('<p class="section-label">Hiperparámetros</p>', unsafe_allow_html=True)
        hyperparams = {}

        if model_name == "Regresión Polinomial":
            degree = st.slider("Grado polinomial (n)", 1, 10, 2)
            hyperparams["degree"] = degree

        elif model_name == "Árbol de Decisión":
            max_depth = st.slider("Profundidad máxima", 1, 20, 5)
            hyperparams["max_depth"] = max_depth

        elif model_name == "SVR":
            kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            C = st.slider("C (Regularización)", 0.01, 100.0, 1.0, step=0.1)
            epsilon = st.slider("Epsilon", 0.001, 2.0, 0.1, step=0.01)
            hyperparams = {"kernel": kernel, "C": C, "epsilon": epsilon}

        elif model_name == "KNN":
            n_neighbors = st.slider("Vecinos (k)", 1, 20, 5)
            hyperparams["n_neighbors"] = n_neighbors

        elif model_name == "Random Forest":
            n_estimators = st.slider("Número de árboles", 10, 500, 100, step=10)
            max_depth_rf = st.slider("Profundidad máxima", 1, 20, 5)
            hyperparams = {"n_estimators": n_estimators, "max_depth": max_depth_rf}

        st.divider()

        # -- Tabla de Datos --
        st.markdown('<p class="section-label">Datos de Medición</p>', unsafe_allow_html=True)
        y_col = "Longitud Sombra (cm)" if is_shadow else "Intensidad (Lux)"

        # Datos por defecto según modo
        if is_shadow:
            default_data = pd.DataFrame({
                "Hora (X)": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                y_col: [45.0, 30.0, 20.0, 12.0, 7.0, 5.0, 7.0, 13.0, 22.0, 32.0, 48.0],
            })
        else:
            default_data = pd.DataFrame({
                "Hora (X)": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                y_col: [120.0, 280.0, 480.0, 650.0, 820.0, 900.0, 870.0, 700.0, 500.0, 300.0, 130.0],
            })

        edited_df = st.data_editor(
            default_data,
            num_rows="dynamic",
            use_container_width=True,
            key=f"data_editor_{mode}",
        )

        # -- Botón Calcular (justo bajo la tabla) --
        calculate = st.button("✨ Calcular Modelo", type="primary", use_container_width=True)

        st.divider()

        # -- Subida de evidencias --
        st.markdown('<p class="section-label">Evidencias (Fotos)</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Sube imágenes del experimento",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            st.session_state.evidence_files = [(f.name, f.read()) for f in uploaded]
            st.success(f"{len(uploaded)} imagen(es) cargada(s)")

        st.divider()

        # -- Notas --
        st.markdown('<p class="section-label">Observaciones</p>', unsafe_allow_html=True)
        notes = st.text_area("Notas del experimento", height=80, label_visibility="collapsed",
                             placeholder="Escribe observaciones sobre las condiciones del experimento...")

    # ─────────────────────────────────────────────────────────────────────────
    # LÓGICA DE CÁLCULO
    # ─────────────────────────────────────────────────────────────────────────
    if calculate:
        # Validación
        df_clean = edited_df.dropna()
        if len(df_clean) < 2:
            st.error("⚠️ Se necesitan al menos 2 puntos de datos válidos para ajustar un modelo.")
        else:
            X = df_clean["Hora (X)"].values
            Y = df_clean[y_col].values
            try:
                result = fit_model(model_name, X, Y, **hyperparams)
                # Derivada si aplica
                deriv_result = None
                if is_shadow and model_name == "Regresión Polinomial":
                    deriv_result = compute_derivative(
                        result["poly_coefficients"], result["x_smooth"]
                    )

                st.session_state.results = {
                    "result": result,
                    "deriv": deriv_result,
                    "X": X,
                    "Y": Y,
                    "model_name": model_name,
                    "mode": mode,
                    "hyperparams": hyperparams,
                    "notes": notes,
                    "y_col": y_col,
                }

                # Guardar en historial
                mode_short = "Sombra" if is_shadow else "Luz"
                save_run(
                    mode=mode_short,
                    model_name=model_name,
                    hyperparams=hyperparams,
                    equation=result.get("equation", ""),
                    metrics=result["metrics"],
                    data_x=X,
                    data_y=Y,
                    notes=notes,
                )

            except ValueError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:
                st.error(f"Error inesperado: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # PANEL DERECHO: Resultados y Gráficas
    # ─────────────────────────────────────────────────────────────────────────
    with col_right:
        if st.session_state.results is not None:
            res = st.session_state.results
            result = res["result"]
            deriv = res["deriv"]
            X = res["X"]
            Y = res["Y"]
            y_col_name = res["y_col"]

            # ── Tarjetas de métricas ──
            eq_col, r2_col, mse_col, rmse_col = st.columns([2, 1, 1, 1])

            with eq_col:
                st.markdown(f"""
                <div class="equation-card">
                    <div class="eq-label">Modelo Matemático</div>
                    <div class="eq-text">{result['equation']}</div>
                </div>
                """, unsafe_allow_html=True)

            with r2_col:
                r2_val = result["metrics"]["R2"]
                r2_pct = max(0, r2_val) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">R² (Precisión)</div>
                    <div class="metric-value metric-r2">{r2_val}</div>
                    <div style="width:100%;background:#d1fae5;height:6px;border-radius:99px;margin-top:0.4rem;overflow:hidden;">
                        <div style="width:{r2_pct}%;background:#10b981;height:100%;border-radius:99px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with mse_col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MSE (Error)</div>
                    <div class="metric-value metric-mse">{result['metrics']['MSE']}</div>
                </div>
                """, unsafe_allow_html=True)

            with rmse_col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value metric-rmse">{result['metrics']['RMSE']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Ecuación LaTeX (si es polinomial) ──
            if res["model_name"] == "Regresión Polinomial" and "poly_coefficients" in result:
                coefs = result["poly_coefficients"]
                degree = len(coefs) - 1
                latex_terms = []
                for i, c in enumerate(coefs):
                    power = degree - i
                    c_r = round(c, 4)
                    if c_r == 0:
                        continue
                    if c_r >= 0 and latex_terms:
                        sign = "+"
                    elif c_r < 0:
                        sign = "-"
                        c_r = abs(c_r)
                    else:
                        sign = ""
                    if power == 0:
                        latex_terms.append(f"{sign}{c_r}")
                    elif power == 1:
                        latex_terms.append(f"{sign}{c_r}x")
                    else:
                        latex_terms.append(f"{sign}{c_r}x^{{{power}}}")
                latex_eq = "y = " + " ".join(latex_terms) if latex_terms else "y = 0"
                st.latex(latex_eq)

            # ── Gráfica Principal ──
            is_shadow_mode = "Sombra" in res["mode"]
            chart_title = "Rastreo de Sombras (Gnomon)" if is_shadow_mode else "Ajuste de Intensidad Solar"
            chart_subtitle = f"{res['model_name']}"

            fig_main = go.Figure()

            # Puntos reales
            fig_main.add_trace(go.Scatter(
                x=X, y=Y,
                mode="markers",
                name="Mediciones",
                marker=dict(
                    size=10,
                    color="#f59e0b",
                    line=dict(width=1.5, color="white"),
                ),
            ))

            # Curva del modelo
            fig_main.add_trace(go.Scatter(
                x=result["x_smooth"],
                y=result["y_smooth"],
                mode="lines",
                name="Modelo",
                line=dict(color="#4f46e5", width=3),
            ))

            fig_main.update_layout(
                title=dict(
                    text=f"<b>{chart_title}</b><br><span style='font-size:13px;color:#64748b;'>{chart_subtitle}</span>",
                    font=dict(size=17, family="Inter, sans-serif", color="#0f172a"),
                ),
                xaxis_title="Hora (h)",
                yaxis_title=y_col_name,
                template="plotly_white",
                height=450,
                paper_bgcolor="white",
                plot_bgcolor="white",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=11, color="#0f172a"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#e2e8f0",
                    borderwidth=1,
                ),
                margin=dict(l=60, r=30, t=80, b=50),
                font=dict(family="Inter, sans-serif", color="#334155"),
            )
            fig_main.update_xaxes(
                gridcolor="#e2e8f0", gridwidth=1,
                linecolor="#cbd5e1", linewidth=1,
                tickfont=dict(color="#475569", size=11),
                title_font=dict(color="#334155"),
            )
            fig_main.update_yaxes(
                gridcolor="#e2e8f0", gridwidth=1,
                linecolor="#cbd5e1", linewidth=1,
                tickfont=dict(color="#475569", size=11),
                title_font=dict(color="#334155"),
            )

            st.plotly_chart(fig_main, use_container_width=True, key="main_chart")

            # ── Gráfica de Derivada (solo Sombra + Polinomial) ──
            fig_deriv = None
            if deriv is not None:
                fig_deriv = go.Figure()
                fig_deriv.add_trace(go.Scatter(
                    x=result["x_smooth"],
                    y=deriv["y_derivative"],
                    mode="lines",
                    name="Velocidad v(x)",
                    line=dict(color="#4f46e5", width=3),
                    fill="tozeroy",
                    fillcolor="rgba(79,70,229,0.08)",
                ))
                # Línea de referencia en y=0
                fig_deriv.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1.5)

                fig_deriv.update_layout(
                    title=dict(
                        text="<b>Velocidad de Variación</b><br>"
                             f"<span style='font-size:13px;color:#64748b;'>Primera derivada: {deriv['equation']}</span>",
                        font=dict(size=17, color="#0f172a", family="Inter, sans-serif"),
                    ),
                    xaxis_title="Hora (h)",
                    yaxis_title="Velocidad (cm/h)",
                    template="plotly_white",
                    height=380,
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    legend=dict(
                        font=dict(color="#0f172a", size=11),
                        bgcolor="rgba(255,255,255,0.9)",
                        bordercolor="#e2e8f0",
                        borderwidth=1,
                    ),
                    margin=dict(l=60, r=30, t=90, b=50),
                    font=dict(family="Inter, sans-serif", color="#334155"),
                )
                fig_deriv.update_xaxes(
                    gridcolor="#e2e8f0", gridwidth=1,
                    linecolor="#cbd5e1", linewidth=1,
                    tickfont=dict(color="#475569", size=11),
                    title_font=dict(color="#334155"),
                )
                fig_deriv.update_yaxes(
                    gridcolor="#e2e8f0", gridwidth=1,
                    linecolor="#cbd5e1", linewidth=1,
                    tickfont=dict(color="#475569", size=11),
                    title_font=dict(color="#334155"),
                    zeroline=True, zerolinecolor="#94a3b8", zerolinewidth=1.5,
                )

                st.plotly_chart(fig_deriv, use_container_width=True, key="deriv_chart")

                # Ecuación derivada en LaTeX
                if "derivative_coefficients" in deriv:
                    d_coefs = deriv["derivative_coefficients"]
                    d_degree = len(d_coefs) - 1
                    lat = []
                    for i, c in enumerate(d_coefs):
                        pw = d_degree - i
                        cr = round(c, 4)
                        if cr == 0:
                            continue
                        if cr >= 0 and lat:
                            s = "+"
                        elif cr < 0:
                            s = "-"
                            cr = abs(cr)
                        else:
                            s = ""
                        if pw == 0:
                            lat.append(f"{s}{cr}")
                        elif pw == 1:
                            lat.append(f"{s}{cr}x")
                        else:
                            lat.append(f"{s}{cr}x^{{{pw}}}")
                    st.latex("v(x) = \\frac{dy}{dx} = " + (" ".join(lat) if lat else "0"))

            # ── Exportar PDF ──
            st.divider()
            st.markdown('<p class="section-label">Exportar Reporte</p>', unsafe_allow_html=True)

            # Generar imágenes de gráficas para PDF
            plot_bytes = fig_main.to_image(format="png", width=1200, height=500, scale=2)
            deriv_bytes = None
            if fig_deriv is not None:
                deriv_bytes = fig_deriv.to_image(format="png", width=1200, height=400, scale=2)

            mode_label = "Rastreo de Sombras" if is_shadow_mode else "Intensidad Lumínica"
            pdf_bytes = generate_pdf(
                mode=mode_label,
                model_name=res["model_name"],
                data_x=X,
                data_y=Y,
                metrics=result["metrics"],
                equation=result.get("equation", ""),
                plot_image_bytes=plot_bytes,
                derivative_image_bytes=deriv_bytes,
                evidence_images=st.session_state.evidence_files or None,
                notes=res.get("notes", ""),
            )

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Descargar Reporte PDF",
                data=pdf_bytes,
                file_name=f"SolarMotion_Reporte_{timestamp_str}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        else:
            # Estado vacío — guía visual
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                        height:60vh;text-align:center;color:#94a3b8;">
                <div style="font-size:4rem;margin-bottom:1rem;">☀️</div>
                <h2 style="font-size:1.5rem;font-weight:700;color:#cbd5e1;margin-bottom:0.5rem;">
                    SolarMotion Tracker
                </h2>
                <p style="max-width:400px;line-height:1.6;">
                    Ingresa tus datos de medición en el panel izquierdo, selecciona un modelo
                    y presiona <strong style="color:#4f46e5;">✨ Calcular Modelo</strong> para
                    visualizar el ajuste.
                </p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# TAB 2 — HISTORIAL
# =============================================================================
with tab_history:
    st.markdown("### 📂 Historial de Ejecuciones")
    st.caption("Cada cálculo se guarda automáticamente para comparación.")

    history = get_history(limit=100)

    if not history:
        st.info("No hay ejecuciones guardadas aún. Realiza un cálculo para comenzar el historial.")
    else:
        # Botón para limpiar historial
        c1, c2 = st.columns([4, 1])
        with c2:
            if st.button("🗑️ Limpiar historial", type="secondary"):
                clear_history()
                st.rerun()

        # Tabla resumen
        hist_df = pd.DataFrame(history)
        display_cols = ["id", "timestamp", "mode", "model_name", "equation", "mse", "rmse", "r2"]
        available_cols = [c for c in display_cols if c in hist_df.columns]
        hist_display = hist_df[available_cols].copy()
        hist_display.columns = [
            c.replace("mse", "MSE").replace("rmse", "RMSE").replace("r2", "R²")
             .replace("timestamp", "Fecha").replace("mode", "Modo")
             .replace("model_name", "Modelo").replace("equation", "Ecuación")
             .replace("id", "ID")
            for c in available_cols
        ]

        st.dataframe(
            hist_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "R²": st.column_config.NumberColumn(format="%.4f"),
                "MSE": st.column_config.NumberColumn(format="%.6f"),
                "RMSE": st.column_config.NumberColumn(format="%.6f"),
            },
        )

        # Tarjetas detalladas
        st.markdown("---")
        st.markdown("#### Detalle de Ejecuciones")
        for run in history[:20]:
            badge_class = "history-badge-sombra" if run["mode"] == "Sombra" else "history-badge-luz"
            r2_color = "#10b981" if (run.get("r2") or 0) > 0.9 else ("#f59e0b" if (run.get("r2") or 0) > 0.7 else "#f43f5e")
            st.markdown(f"""
            <div class="history-item">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.4rem;">
                    <span class="{badge_class}">{run['mode']}</span>
                    <span style="font-size:0.7rem;color:#94a3b8;">{run['timestamp']}</span>
                </div>
                <div style="font-size:0.9rem;font-weight:700;color:#0f172a;margin-bottom:0.2rem;">
                    {run['model_name']}
                </div>
                <div style="font-family:'Fira Code',monospace;font-size:0.75rem;color:#4f46e5;margin-bottom:0.4rem;">
                    {run.get('equation', '')}
                </div>
                <div style="display:flex;gap:1.5rem;font-size:0.7rem;color:#64748b;">
                    <span>R²: <strong style="color:{r2_color};">{run.get('r2', 'N/A')}</strong></span>
                    <span>MSE: <strong>{run.get('mse', 'N/A')}</strong></span>
                    <span>RMSE: <strong>{run.get('rmse', 'N/A')}</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
