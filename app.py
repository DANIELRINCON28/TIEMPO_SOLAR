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

# ─── Helpers de tiempo HH.MM ─────────────────────────────────────────────────
def _hhmm_to_dec(v):
    """HH.MM (ej: 10.05 = 10h 05min) → horas decimales (10.0833)."""
    h = int(v)
    m = round((v - h) * 100)
    return h + m / 60.0

def _dec_to_label(dec_h):
    """Horas decimales → etiqueta 'HH:MM' (ej: 10.0833 → '10:05')."""
    h = int(dec_h)
    m = round((dec_h - h) * 60)
    if m == 60:
        h += 1; m = 0
    return f"{h}:{m:02d}"

def _dec_to_dt(dec_h):
    """Horas decimales → string ISO datetime (fecha ficticia 2000-01-01).
    Plotly usa esto para espaciar el eje X por tiempo real."""
    from datetime import datetime
    h = int(dec_h)
    m_frac = (dec_h - h) * 60
    m = int(m_frac)
    s = round((m_frac - m) * 60)
    if s == 60: m += 1; s = 0
    if m == 60: h += 1; m = 0
    return datetime(2000, 1, 1, min(h, 23), m, s).isoformat()

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

        # Textos por defecto según modo
        if is_shadow:
            _default_text = (
                "7.0,45.0\n8.0,30.0\n9.0,20.0\n10.0,12.0\n11.0,7.0\n"
                "12.0,5.0\n13.0,7.0\n14.0,13.0\n15.0,22.0\n16.0,32.0\n17.0,48.0"
            )
        else:
            _default_text = (
                "7.0,120.0\n8.0,280.0\n9.0,480.0\n10.0,650.0\n11.0,820.0\n"
                "12.0,900.0\n13.0,870.0\n14.0,700.0\n15.0,500.0\n16.0,300.0\n17.0,130.0"
            )

        _txt_key = f"raw_text_{mode}"
        if _txt_key not in st.session_state:
            st.session_state[_txt_key] = _default_text

        _hhmm_key = f"hhmm_{mode}"
        _use_hhmm = st.session_state.get(_hhmm_key, False)

        st.markdown(
            '<p style="font-size:12px;color:#64748b;margin-bottom:4px;">'
            'Pega o escribe los datos: <code>hora,valor</code> — una fila por línea</p>',
            unsafe_allow_html=True,
        )
        raw_text = st.text_area(
            "datos_raw",
            value=st.session_state[_txt_key],
            height=220,
            label_visibility="collapsed",
            placeholder="9.50,0\n9.55,8\n10.05,20.5\n...",
            key=f"textarea_{mode}",
        )
        st.session_state[_txt_key] = raw_text

        _use_hhmm = st.checkbox(
            "⏱ Formato HH.MM — el decimal son minutos (ej: 10.05 = 10h 05min)",
            value=_use_hhmm,
            key=f"cb_hhmm_{mode}",
            help="Actívalo si tu hora usa punto como separador de minutos (0–59), no como fracción decimal.",
        )
        st.session_state[_hhmm_key] = _use_hhmm

        # -- Parsear el texto a DataFrame --
        def _parse_raw(text, col_y, hhmm=False):
            rows = []
            errors = []
            for i, line in enumerate(text.strip().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                # Aceptar coma o punto y coma como separador
                sep = ";" if ";" in line else ","
                parts = line.split(sep)
                if len(parts) < 2:
                    errors.append(f"Línea {i}: «{line}» — necesita dos valores")
                    continue
                try:
                    x_raw = float(parts[0].strip())
                    y_val = float(parts[1].strip())
                    x_dec = _hhmm_to_dec(x_raw) if hhmm else x_raw
                    x_lbl = _dec_to_label(x_dec) if hhmm else f"{x_raw:.2f}h"
                    rows.append({"Hora (X)": x_dec, "_lbl": x_lbl, col_y: y_val})
                except ValueError:
                    errors.append(f"Línea {i}: «{line}» — valor no numérico")
            df_out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Hora (X)", "_lbl", col_y])
            return df_out, errors

        _df_full, _parse_errors = _parse_raw(raw_text, y_col, hhmm=_use_hhmm)
        edited_df = _df_full.drop(columns=["_lbl"], errors="ignore")

        if _parse_errors:
            for _e in _parse_errors:
                st.warning(_e)

        # Vista previa compacta
        if not edited_df.empty:
            st.markdown(
                f'<p style="font-size:12px;color:#64748b;margin-top:6px;">'
                f'✅ {len(edited_df)} punto(s) leído(s)</p>',
                unsafe_allow_html=True,
            )

        # -- Botones --
        _col_reset, _col_calc = st.columns([1, 2])
        with _col_reset:
            if st.button("↺ Restaurar ejemplo", use_container_width=True,
                         help="Vuelve a los datos de ejemplo"):
                st.session_state[_txt_key] = _default_text
                st.rerun()
        with _col_calc:
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
            _hhmm_active = st.session_state.get(f"hhmm_{mode}", False)
            x_labels_str = [_dec_to_label(v) for v in X.tolist()]
            x_dt_str = [_dec_to_dt(v) for v in X.tolist()]
            try:
                result = fit_model(model_name, X, Y, **hyperparams)
                # Derivada si aplica
                deriv_result = None
                if is_shadow and model_name == "Regresión Polinomial":
                    deriv_result = compute_derivative(
                        result["poly_coefficients"], result["x_smooth"]
                    )

                # Convertir todos los numpy arrays a listas Python para
                # compatibilidad con Plotly 6.x y serialización de session_state
                def _tolist(v):
                    import numpy as np
                    return v.tolist() if isinstance(v, np.ndarray) else v

                result_clean = {
                    k: _tolist(v) for k, v in result.items()
                    if k not in ("model", "_scaler_x", "_scaler_y")
                }

                deriv_clean = None
                if deriv_result is not None:
                    deriv_clean = {k: _tolist(v) for k, v in deriv_result.items()}

                xs_dt_str = [_dec_to_dt(v) for v in result_clean["x_smooth"]]

                st.session_state.results = {
                    "result": result_clean,
                    "deriv": deriv_clean,
                    "X": X.tolist(),
                    "Y": Y.tolist(),
                    "X_labels": x_labels_str,
                    "X_dt": x_dt_str,
                    "xs_dt": xs_dt_str,
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
            x_labels = res.get("X_labels", [_dec_to_label(v) for v in X])
            X_dt = res.get("X_dt", [_dec_to_dt(v) for v in X])
            xs_dt = res.get("xs_dt", [_dec_to_dt(v) for v in (result["x_smooth"] if isinstance(result["x_smooth"], list) else list(result["x_smooth"]))])

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

            # Datos numéricos para ML ya están en res; para la gráfica usamos datetime
            py = list(Y) if not isinstance(Y, list) else Y
            ys = result["y_smooth"] if isinstance(result["y_smooth"], list) else list(result["y_smooth"])

            # Puntos reales — eje X como datetime para espaciado real
            fig_main.add_trace(go.Scatter(
                x=X_dt, y=py,
                mode="markers",
                name="Mediciones",
                marker=dict(
                    size=11,
                    color="#f59e0b",
                    line=dict(width=2, color="white"),
                    symbol="circle",
                ),
            ))

            # Curva del modelo — eje X como datetime
            fig_main.add_trace(go.Scatter(
                x=xs_dt, y=ys,
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
                type="date",
                tickformat="%H:%M",
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
                # Convertir a listas puras para Plotly 6.x
                dyd = deriv["y_derivative"] if isinstance(deriv["y_derivative"], list) else list(deriv["y_derivative"])
                fig_deriv.add_trace(go.Scatter(
                    x=xs_dt,
                    y=dyd,
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
                    type="date",
                    tickformat="%H:%M",
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
