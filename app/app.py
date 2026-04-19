import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import io, base64, time
import os
import requests

# ════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OralScan AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════
MODEL_PATH = Path(__file__).parent.parent / "modelos" / "mobilenet_con_aug" / "best.pt"
DEVICE     = torch.device("cpu")   # CPU en cloud
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
API_URL    = None

# Métricas del mejor modelo (MobileNetV2 con DA)
MODEL_METRICS = {
    "Accuracy": 0.9722,
    "AUC-ROC":  0.9946,
    "F1-Score": 0.9733,
    "Precision":0.9712,
    "Recall":   0.9754,
}


def load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_env_file()
API_URL = os.getenv("API_URL")

# ════════════════════════════════════════════════════════════
#  CSS — diseño "Clinical Luminary" de Stitch
# ════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── fondo global ── */
.stApp { background-color: #0b1326; }
section[data-testid="stSidebar"] { background-color: #131b2e !important; border-right: none; }

/* ── sidebar ── */
.sidebar-logo {
    display: flex; align-items: center; gap: 12px;
    padding: 24px 0 8px 0;
}
.sidebar-logo-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #06b6d4, #0e7490);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; box-shadow: 0 0 20px rgba(6,182,212,0.3);
}
.sidebar-title { color: #dae2fd; font-size: 18px; font-weight: 700; line-height: 1.1; }
.sidebar-subtitle { color: #4cd7f6; font-size: 11px; font-weight: 500;
    letter-spacing: 0.05em; text-transform: uppercase; }
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; border-radius: 8px; margin: 3px 0;
    color: #bcc9cd; font-size: 14px; font-weight: 500;
    cursor: pointer; transition: all 0.2s; text-decoration: none;
    border-left: 2px solid transparent;
}
.nav-item.active {
    background: #222a3d; color: #4cd7f6;
    border-left: 2px solid #4cd7f6;
}
.nav-item:hover { background: #171f33; color: #dae2fd; }

/* ── upload zone ── */
.upload-zone {
    background: linear-gradient(135deg, rgba(6,182,212,0.05) 0%, #171f33 100%);
    border: 1.5px dashed rgba(134,147,151,0.4);
    border-radius: 16px; padding: 60px 40px;
    text-align: center; transition: all 0.3s;
}
.upload-icon { font-size: 48px; margin-bottom: 16px; }
.upload-title { color: #dae2fd; font-size: 20px; font-weight: 600; margin-bottom: 8px; }
.upload-sub { color: #869397; font-size: 14px; }

/* ── resultado cards ── */
.result-card {
    background: #222a3d; border-radius: 12px; padding: 20px;
    margin-bottom: 12px;
}
.metric-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: #2d3449; border-radius: 10px;
    padding: 16px 12px; text-align: center;
}
.metric-value { color: #4cd7f6; font-size: 22px; font-weight: 700; }
.metric-label { color: #869397; font-size: 11px; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

/* ── diagnosis badge ── */
.diagnosis-cancer {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.1));
    border: 1px solid rgba(239,68,68,0.3); border-radius: 12px;
    padding: 16px 20px; text-align: center; margin-bottom: 12px;
}
.diagnosis-normal {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(21,128,61,0.1));
    border: 1px solid rgba(34,197,94,0.3); border-radius: 12px;
    padding: 16px 20px; text-align: center; margin-bottom: 12px;
}
.diagnosis-text-cancer { color: #f87171; font-size: 26px; font-weight: 800; }
.diagnosis-text-normal { color: #4ade80; font-size: 26px; font-weight: 800; }
.diagnosis-sub { color: #bcc9cd; font-size: 13px; margin-top: 4px; }

/* ── progress bar custom ── */
.conf-bar-container { margin: 8px 0; }
.conf-bar-label { display: flex; justify-content: space-between;
    color: #bcc9cd; font-size: 12px; margin-bottom: 4px; }
.conf-bar-bg { background: #2d3449; border-radius: 100px; height: 8px; }
.conf-bar-fill { height: 8px; border-radius: 100px;
    background: linear-gradient(90deg, #0e7490, #4cd7f6); }

/* ── disclaimer ── */
.disclaimer {
    background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.2);
    border-radius: 10px; padding: 14px 18px; margin-top: 24px;
    color: #fbbf24; font-size: 13px; line-height: 1.5;
}

/* ── section header ── */
.section-title {
    color: #4cd7f6; font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 14px;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 16px; max-width: 1200px; }
/* Botón de toggle del sidebar siempre visible */
[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important;
    background: #131b2e !important; border-radius: 0 8px 8px 0;
    border: 1px solid #2d3449 !important;
}

/* ── file uploader override ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] section {
    background: linear-gradient(135deg, rgba(6,182,212,0.05) 0%, #171f33 100%) !important;
    border: 1.5px dashed rgba(134,147,151,0.4) !important;
    border-radius: 16px !important;
    padding: 40px !important;
}
[data-testid="stFileUploader"] section label {
    color: #dae2fd !important;
}

/* about model table */
.perf-table { width: 100%; border-collapse: collapse; }
.perf-table th { color: #4cd7f6; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.05em; padding: 8px 12px; border-bottom: 1px solid #2d3449; }
.perf-table td { color: #dae2fd; font-size: 14px; padding: 10px 12px;
    border-bottom: 1px solid #171f33; }
.perf-table tr:hover td { background: #222a3d; }

/* scanline animation */
@keyframes scanline {
    0%   { top: 0; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}
.scanline-container { position: relative; overflow: hidden; border-radius: 12px; }
.scanline {
    position: absolute; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #4cd7f6, transparent);
    box-shadow: 0 0 12px #4cd7f6;
    animation: scanline 1.8s ease-in-out infinite;
    z-index: 10;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  MODEL LOADING
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    model = models.mobilenet_v2(weights=None)
    # Arquitectura real del checkpoint: Dropout → Linear(1280,256) → ReLU → Dropout → Linear(256,1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
    )
    state = torch.load(str(MODEL_PATH), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return tfm(img).unsqueeze(0)


def predict(model, tensor: torch.Tensor):
    with torch.no_grad():
        logit = model(tensor).squeeze(-1)
        prob  = torch.sigmoid(logit).item()
    return prob


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def predict_with_api(image: Image.Image) -> float:
    if not API_URL:
        raise RuntimeError("API_URL no está configurada")

    files = {
        "file": ("oral_image.jpg", image_to_jpeg_bytes(image), "image/jpeg"),
    }
    endpoint = f"{API_URL.rstrip('/')}/predict"
    response = requests.post(endpoint, files=files, timeout=45)
    if response.status_code != 200:
        raise RuntimeError(f"La API respondió {response.status_code}: {response.text}")

    payload = response.json()
    if "probability_cancer" not in payload:
        raise RuntimeError("Respuesta de API sin campo 'probability_cancer'")
    return float(payload["probability_cancer"])


def get_prediction_probability(model, tensor: torch.Tensor, image: Image.Image) -> float:
    if API_URL:
        return predict_with_api(image)
    if model is None:
        raise RuntimeError("Modelo local no inicializado")
    return float(predict(model, tensor))


# ════════════════════════════════════════════════════════════
#  COMPONENTS
# ════════════════════════════════════════════════════════════
def gauge_chart(prob: float, is_cancer: bool):
    color = "#ef4444" if is_cancer else "#22c55e"
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(prob * 100, 1),
        number= {"suffix": "%", "font": {"size": 42, "color": color, "family": "Inter"}},
        gauge = {
            "axis": {"range": [0, 100], "tickwidth": 0,
                     "tickcolor": "#3d494c", "tickfont": {"color": "#869397", "size": 11}},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#2d3449",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#1a2640"},
                {"range": [40, 65], "color": "#1e2e44"},
                {"range": [65, 100],"color": "#221a2e"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75, "value": prob * 100,
            },
        },
        title = {"text": "PROBABILIDAD DE CÁNCER",
                 "font": {"size": 12, "color": "#869397", "family": "Inter"}},
    ))
    fig.update_layout(
        height=250, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#1d2538", plot_bgcolor="#1d2538", font_family="Inter",
    )
    return fig


def confidence_bar(label: str, value: float, color: str = "#4cd7f6"):
    pct = int(value * 100)
    return f"""
    <div class="conf-bar-container">
        <div class="conf-bar-label"><span>{label}</span><span>{pct}%</span></div>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct}%; background:linear-gradient(90deg,#0e7490,{color});"></div>
        </div>
    </div>
    """


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🔬</div>
        <div>
            <div class="sidebar-title">OralScan AI</div>
            <div class="sidebar-subtitle">Clasificación de Cáncer Bucal</div>
        </div>
    </div>
    <hr style="border:none; border-top:1px solid #2d3449; margin:16px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        label="Navegación",
        options=["Diagnóstico", "Sobre el Modelo", "Información"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="position:fixed; bottom:24px; left:0; width:240px; padding:0 16px;">
        <div style="color:#3d494c; font-size:11px; text-align:center;">
            MobileNetV2 · PyTorch<br>
            <span style="color:#4cd7f6;">v1.0</span> · CPU Inference
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: DIAGNÓSTICO
# ════════════════════════════════════════════════════════════
if page == "Diagnóstico":
    st.markdown("""
    <div style="margin-bottom:28px;">
        <h1 style="color:#dae2fd; font-size:28px; font-weight:700; margin:0;">Análisis de Imagen Oral</h1>
        <p style="color:#869397; font-size:14px; margin-top:6px;">
            Sube una imagen intraoral para obtener una clasificación asistida por IA.
        </p>
    </div>
    """, unsafe_allow_html=True)

    model = None
    if API_URL:
        st.caption(f"Modo API activo: `{API_URL}`")
    else:
        # ── Carga del modelo local ───────────────────────────
        with st.spinner("Cargando modelo..."):
            try:
                model = load_model()
                st.success("✅ Modelo cargado — MobileNetV2 (fine-tuning con datos aumentados)")
            except Exception as e:
                st.error(f"❌ Error al cargar el modelo: {e}")
                st.stop()

    # ── Uploader ──────────────────────────────────────────────
    uploaded = st.file_uploader(
        label="Arrastra tu imagen aquí o haz click para seleccionar",
        type=["jpg", "jpeg", "png"],
        help="Formatos soportados: JPG, JPEG, PNG. Tamaño máximo: 200 MB.",
    )

    if uploaded is None:
        st.markdown("""
        <div style="margin-top:8px; margin-bottom:4px;">
            <div class="section-title">Consejos para tomar una buena foto intraoral</div>
        </div>
        """, unsafe_allow_html=True)

        tip_col1, tip_col2 = st.columns(2, gap="medium")

        with tip_col1:
            st.markdown("""
            <div style="background:#222a3d; border-radius:10px; padding:18px 20px; margin-bottom:14px;">
                <div style="color:#4cd7f6; font-size:12px; font-weight:600; letter-spacing:0.06em; margin-bottom:10px;">POSICION Y APERTURA</div>
                <div style="color:#bcc9cd; font-size:13.5px; line-height:1.85;">
                    &bull; Abre la boca lo <b>más posible</b><br>
                    &bull; Inclina la cabeza hacia una fuente de luz<br>
                    &bull; Usa un espejo para zonas posteriores<br>
                    &bull; Centra la zona sospechosa en el encuadre
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#222a3d; border-radius:10px; padding:18px 20px;">
                <div style="color:#4cd7f6; font-size:12px; font-weight:600; letter-spacing:0.06em; margin-bottom:10px;">DISTANCIA Y ENCUADRE</div>
                <div style="color:#bcc9cd; font-size:13.5px; line-height:1.85;">
                    &bull; Distancia recomendada: <b>10&ndash;20 cm</b> de la boca<br>
                    &bull; Encuadre horizontal, no diagonal<br>
                    &bull; Incluye tejido circundante como referencia<br>
                    &bull; No dejes que labios o dedos tapen la zona
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tip_col2:
            st.markdown("""
            <div style="background:#222a3d; border-radius:10px; padding:18px 20px; margin-bottom:14px;">
                <div style="color:#4cd7f6; font-size:12px; font-weight:600; letter-spacing:0.06em; margin-bottom:10px;">ILUMINACION Y ENFOQUE</div>
                <div style="color:#bcc9cd; font-size:13.5px; line-height:1.85;">
                    &bull; Activa el <b>flash del celular</b> o usa linterna<br>
                    &bull; Evita contraluz (no foto hacia una ventana)<br>
                    &bull; Toca la pantalla para enfocar en la lesión<br>
                    &bull; Toma varias fotos y elige la más nítida
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div style="background:#222a3d; border-radius:10px; padding:18px 20px;">
                <div style="color:#4cd7f6; font-size:12px; font-weight:600; letter-spacing:0.06em; margin-bottom:10px;">QUE DEBE MOSTRAR LA IMAGEN</div>
                <div style="color:#bcc9cd; font-size:13.5px; line-height:1.85;">
                    &bull; Lengua, encías, paladar o mejilla según la zona<br>
                    &bull; Manchas blancas, rojas o úlceras visibles<br>
                    &bull; Sin filtros ni recortes extremos<br>
                    &bull; <b>En color real</b> (no blanco y negro)
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:16px; color:#3d494c; font-size:12px; border-top:1px solid #2d3449; padding-top:13px;">
            El modelo fue entrenado con imágenes clínicas intraorales en condiciones similares.
            Imágenes de baja calidad pueden afectar la precisión del análisis.
        </div>
        """, unsafe_allow_html=True)
    else:
        img = Image.open(uploaded).convert("RGB")

        col_img, col_res = st.columns([1, 1], gap="large")

        # ── Columna imagen ────────────────────────────────────
        with col_img:
            st.markdown('<div class="section-title">Imagen analizada</div>', unsafe_allow_html=True)
            st.image(img, caption=uploaded.name, use_container_width=True)

        # ── Columna resultados ────────────────────────────────
        with col_res:
            st.markdown('<div class="section-title">Resultados del análisis</div>', unsafe_allow_html=True)

            # Animación "scanline" durante inferencia
            scan_ph = st.empty()
            scan_ph.markdown("""
            <div class="scanline-container" style="background:#171f33; height:60px; margin-bottom:12px;">
                <div class="scanline"></div>
                <div style="text-align:center; padding-top:18px; color:#4cd7f6; font-size:13px;">
                    Analizando imagen…
                </div>
            </div>
            """, unsafe_allow_html=True)

            tensor    = preprocess(img)
            cancer_p  = get_prediction_probability(model, tensor, img)
            normal_p  = 1.0 - cancer_p
            is_cancer = cancer_p >= 0.5
            time.sleep(0.4)  # pequeña pausa para UX
            scan_ph.empty()

            # Diagnosis badge
            if is_cancer:
                st.markdown(f"""
                <div class="diagnosis-cancer">
                    <div class="diagnosis-text-cancer">⚠️ CÁNCER DETECTADO</div>
                    <div class="diagnosis-sub">El modelo clasifica esta imagen como positiva para cáncer bucal</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="diagnosis-normal">
                    <div class="diagnosis-text-normal">✅ NORMAL</div>
                    <div class="diagnosis-sub">El modelo clasifica esta imagen como tejido normal</div>
                </div>
                """, unsafe_allow_html=True)

            # Gauge
            st.plotly_chart(
                gauge_chart(cancer_p, is_cancer),
                use_container_width=True, config={"displayModeBar": False}
            )

            # Barras de confianza
            cancer_color = "#ef4444" if is_cancer else "#4cd7f6"
            st.markdown(
                confidence_bar("Probabilidad de Cáncer", cancer_p, "#ef4444") +
                confidence_bar("Probabilidad Normal",    normal_p, "#22c55e"),
                unsafe_allow_html=True,
            )

            # Métricas del modelo
            st.markdown("""
            <div style="margin-top:20px;">
                <div class="section-title">Rendimiento del modelo</div>
                <div class="metric-grid">
            """, unsafe_allow_html=True)

            m_cols = st.columns(3)
            metrics_display = [
                ("Accuracy", f"{MODEL_METRICS['Accuracy']*100:.1f}%"),
                ("AUC-ROC",  f"{MODEL_METRICS['AUC-ROC']:.4f}"),
                ("F1-Score", f"{MODEL_METRICS['F1-Score']*100:.1f}%"),
            ]
            for col, (label, val) in zip(m_cols, metrics_display):
                col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Aviso importante:</strong> Esta herramienta es un sistema de apoyo diagnóstico
        basado en inteligencia artificial. <strong>No reemplaza la consulta médica profesional.</strong>
        Los resultados deben ser interpretados por un especialista en salud bucodental.
        No tome decisiones clínicas basadas únicamente en esta herramienta.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: SOBRE EL MODELO
# ════════════════════════════════════════════════════════════
elif page == "Sobre el Modelo":
    st.markdown("""
    <h1 style="color:#dae2fd; font-size:28px; font-weight:700; margin-bottom:6px;">Sobre el Modelo</h1>
    <p style="color:#869397; font-size:14px; margin-bottom:28px;">
        Arquitectura, entrenamiento y métricas comparativas de los modelos evaluados.
    </p>
    """, unsafe_allow_html=True)

    # ── Modelo seleccionado ───────────────────────────────────
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="result-card">
            <div class="section-title">Modelo Seleccionado</div>
            <div style="color:#4cd7f6; font-size:22px; font-weight:700; margin-bottom:8px;">
                MobileNetV2
            </div>
            <div style="color:#bcc9cd; font-size:14px; line-height:1.7;">
                <b>Estrategia:</b> Transfer Learning + Fine-Tuning (2 fases)<br>
                <b>Dataset:</b> Con Data Augmentation (10 técnicas)<br>
                <b>Parámetros:</b> ~3.4M (backbone) + 1.3K (clasificador)<br>
                <b>Input:</b> 224 × 224 píxeles (RGB)<br>
                <b>Loss:</b> BCEWithLogitsLoss + class weights<br>
                <b>Optimizador:</b> Adam + ReduceLROnPlateau<br>
                <b>Hardware:</b> NVIDIA RTX 5070 Ti (CUDA 12.8)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Métricas en Test Set</div>', unsafe_allow_html=True)
        for label, val in MODEL_METRICS.items():
            bar_val = val
            pct     = int(bar_val * 100)
            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div class="conf-bar-label" style="display:flex; justify-content:space-between; color:#bcc9cd; font-size:13px; margin-bottom:5px;">
                    <span>{label}</span><span style="color:#4cd7f6; font-weight:600;">{bar_val:.4f}</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Comparativa de experimentos ───────────────────────────
    st.markdown("""
    <div style="margin-top:32px; margin-bottom:16px;">
        <div class="section-title">Comparativa de los 6 experimentos</div>
    </div>
    """, unsafe_allow_html=True)

    experiments = [
        {"Modelo": "CNN Scratch",     "Dataset": "Sin Aug",  "Accuracy": "88.3%", "AUC": "0.9601", "F1": "0.8812"},
        {"Modelo": "CNN Scratch",     "Dataset": "Con Aug",  "Accuracy": "90.1%", "AUC": "0.9698", "F1": "0.9023"},
        {"Modelo": "EfficientNetB0",  "Dataset": "Sin Aug",  "Accuracy": "93.4%", "AUC": "0.9845", "F1": "0.9321"},
        {"Modelo": "EfficientNetB0",  "Dataset": "Con Aug",  "Accuracy": "95.8%", "AUC": "0.9914", "F1": "0.9578"},
        {"Modelo": "MobileNetV2",     "Dataset": "Sin Aug",  "Accuracy": "94.1%", "AUC": "0.9901", "F1": "0.9411"},
        {"Modelo": "MobileNetV2 🏆",  "Dataset": "Con Aug",  "Accuracy": "97.2%", "AUC": "0.9946", "F1": "0.9733"},
    ]

    header = "| Modelo | Dataset | Accuracy | AUC-ROC | F1-Score |\n|---|---|---|---|---|\n"
    rows   = "".join(
        f"| {e['Modelo']} | {e['Dataset']} | {e['Accuracy']} | {e['AUC']} | {e['F1']} |\n"
        for e in experiments
    )
    st.markdown(header + rows)

    # ── Sobre el dataset ──────────────────────────────────────
    st.markdown("""
    <div class="result-card" style="margin-top:24px;">
        <div class="section-title">Dataset</div>
        <div style="color:#bcc9cd; font-size:14px; line-height:1.8;">
            <b>Fuente:</b> Oral Cancer Images Dataset (Kaggle)<br>
            <b>Clases:</b> Cancer (positivo) · Normal (negativo)<br>
            <b>División:</b> 70% entrenamiento / 15% validación / 15% test<br>
            <b>Data Augmentation:</b> Rotación, flip H/V, zoom, brillo, contraste,
            saturación, ruido gaussiano, CLAHE, recorte aleatorio, escalado.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE: INFORMACIÓN
# ════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <h1 style="color:#dae2fd; font-size:28px; font-weight:700; margin-bottom:6px;">Información</h1>
    <p style="color:#869397; font-size:14px; margin-bottom:28px;">
        Contexto clínico y uso responsable de la herramienta.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="result-card">
            <div class="section-title">Sobre el Cáncer Bucal</div>
            <div style="color:#bcc9cd; font-size:14px; line-height:1.8;">
                El <b style="color:#dae2fd;">cáncer bucal</b> representa el 3% de todos los cánceres a nivel mundial.
                La detección temprana aumenta la tasa de supervivencia a 5 años del 50% al <b style="color:#4cd7f6;">83%</b>.
                <br><br>
                Los signos clínicos incluyen lesiones blancas (leucoplasia), lesiones rojas (eritroplasia),
                úlceras persistentes y cambios en la textura de la mucosa oral.
                <br><br>
                <b style="color:#dae2fd;">Factores de riesgo principales:</b><br>
                • Consumo de tabaco y alcohol<br>
                • Infección por VPH (subtipo 16 y 18)<br>
                • Exposición solar crónica (labio)<br>
                • Mala higiene oral crónica
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="result-card">
            <div class="section-title">Sobre OralScan AI</div>
            <div style="color:#bcc9cd; font-size:14px; line-height:1.8;">
                OralScan AI es un sistema de clasificación de imágenes orales basado en
                <b style="color:#4cd7f6;">MobileNetV2</b> con Transfer Learning, entrenado sobre un
                dataset de imágenes clínicas.
                <br><br>
                <b style="color:#dae2fd;">Tecnologías utilizadas:</b><br>
                • PyTorch 2.x · Torchvision<br>
                • Streamlit · Plotly<br>
                • Data Augmentation (10 técnicas)<br>
                • Score-CAM para explicabilidad
                <br><br>
                <b style="color:#dae2fd;">Repositorio:</b><br>
                <a href="https://github.com/ErnestoSCL/clasificacion-cancer-bucal"
                   style="color:#4cd7f6;">
                    github.com/ErnestoSCL/clasificacion-cancer-bucal
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer" style="margin-top:20px;">
        <b>⚠️ Aviso Legal y Ético</b><br>
        Esta herramienta ha sido desarrollada con fines académicos e investigativos.
        <b>No está certificada para uso diagnóstico clínico.</b>
        Cualquier hallazgo debe ser confirmado por un profesional de salud bucodental calificado.
        Los desarrolladores no se hacen responsables por decisiones médicas basadas en los resultados
        de este sistema.
    </div>
    """, unsafe_allow_html=True)
