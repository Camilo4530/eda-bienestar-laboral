"""
Dashboard Interactivo — Bienestar Laboral y Salud Mental Organizacional
Analisis Exploratorio de Datos · Colombia · 2026
Uso: streamlit run dashboard_bienestar_v2.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr

# ── Página ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bienestar Laboral · EDA Colombia",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta neón ───────────────────────────────────────────────────────────
NEON = {
    "cyan":   "#00d4ff",
    "purple": "#a855f7",
    "pink":   "#ff6b9d",
    "green":  "#00f5a0",
    "orange": "#ff8c42",
    "yellow": "#ffd166",
    "blue":   "#4d9fff",
    "red":    "#ff4757",
}
NEON_LIST = list(NEON.values())

BG_DARK  = "#0a0e1a"
BG_CARD  = "#111827"
BG_CARD2 = "#0f172a"
BORDER   = "#1e293b"
TEXT     = "#e2e8f0"
MUTED    = "#64748b"

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

/* Fondo */
.stApp {{ background: {BG_DARK} !important; }}
.main .block-container {{ padding: 16px 32px 32px 32px; max-width: 100%; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0d1225 0%, #111827 100%) !important;
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
section[data-testid="stSidebar"] .stSelectbox label {{
    color: {NEON['cyan']} !important;
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 1.2px;
}}

/* Hero */
.hero {{
    background: linear-gradient(135deg, #0d1225 0%, #0f1f3d 45%, #1a0a2e 100%);
    border: 1px solid {BORDER};
    border-radius: 18px; padding: 32px 40px; margin-bottom: 24px;
    position: relative; overflow: hidden;
    box-shadow: 0 0 60px rgba(0,212,255,0.06), 0 0 120px rgba(168,85,247,0.04);
}}
.hero::before {{
    content: '';
    position: absolute; top: -100px; right: -80px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.1) 0%, transparent 65%);
    border-radius: 50%;
}}
.hero::after {{
    content: '';
    position: absolute; bottom: -80px; left: 20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(168,85,247,0.08) 0%, transparent 65%);
    border-radius: 50%;
}}
.hero-title {{
    font-size: 28px; font-weight: 900; color: #fff !important;
    margin: 0 0 6px 0; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ffffff, {NEON['cyan']});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.hero-sub {{ font-size: 14px; color: #7ea8cc !important; margin: 0; }}
.hero-pill {{
    display: inline-block; margin-top: 14px;
    background: rgba(0,212,255,0.1); color: {NEON['cyan']} !important;
    border: 1px solid rgba(0,212,255,0.3); padding: 5px 18px;
    border-radius: 20px; font-size: 12px; font-weight: 600;
}}

/* Tarjetas KPI */
.kpi {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 14px; padding: 20px 16px; text-align: center;
    position: relative; overflow: hidden;
}}
.kpi::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--kpi-color, {NEON['cyan']});
    box-shadow: 0 0 12px var(--kpi-color, {NEON['cyan']});
}}
.kpi-label {{ font-size: 9px; color: {MUTED} !important; font-weight: 700;
              text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 8px; }}
.kpi-val   {{ font-size: 30px; font-weight: 900; color: #fff !important; line-height: 1; margin-bottom: 5px; }}
.kpi-sub   {{ font-size: 11px; color: {MUTED} !important; }}

/* Tarjetas de gráfica */
.chart-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 20px 20px 12px 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.4);
    position: relative; overflow: hidden;
}}
.chart-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
}}
.chart-title {{
    font-size: 13px; font-weight: 700; color: {TEXT} !important;
    margin: 0 0 2px 0; letter-spacing: 0.2px;
}}
.chart-sub {{
    font-size: 11px; color: {MUTED} !important; margin: 0 0 14px 0;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {BG_CARD}; border-radius: 12px; padding: 4px;
    border: 1px solid {BORDER}; gap: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px; font-weight: 600; font-size: 13px;
    color: {MUTED} !important; padding: 8px 20px; background: transparent;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(0,212,255,0.15), rgba(168,85,247,0.15)) !important;
    color: {NEON['cyan']} !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
}}

/* Selectbox */
.stSelectbox > div > div {{
    background: {BG_CARD2} !important; border-color: {BORDER} !important;
    color: {TEXT} !important; border-radius: 8px !important;
}}
.stSelectbox label {{ color: {MUTED} !important; font-size: 12px !important; font-weight: 600 !important; }}

/* Textos generales */
h1, h2, h3, p, .stMarkdown, .stMarkdown p {{ color: {TEXT} !important; }}
h3 {{ color: {NEON['cyan']} !important; font-size: 15px !important; }}
[data-testid="stMetricValue"] {{ color: #fff !important; font-weight: 900 !important; }}
[data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: {BG_DARK}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: {NEON['cyan']}; }}

hr {{ border: none; height: 1px;
     background: linear-gradient(90deg, transparent, {BORDER}, transparent); margin: 20px 0; }}
</style>
""", unsafe_allow_html=True)


# ── Helper: aplicar tema oscuro a figuras ─────────────────────────────────
def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter", size=12),
        title_text="",
        legend=dict(
            bgcolor="rgba(17,24,39,0.9)",
            bordercolor=BORDER, borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=16, r=16, t=16, b=16),
        height=height,
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=11))
    return fig


def card(title, subtitle=""):
    """Render chart card header via markdown."""
    sub_html = f'<div class="chart-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="chart-card">
        <div class="chart-title">{title}</div>
        {sub_html}
    """, unsafe_allow_html=True)


def end_card():
    st.markdown("</div>", unsafe_allow_html=True)


# ── Carga de datos ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("bienestar_laboral_EDA.xlsx")
    EA = {"Nunca":1,"Rara vez":2,"Algunas veces":3,"Frecuentemente":4,"Siempre":5}
    EB = {"Nunca":1,"Raramente":2,"Algunas veces":3,"A menudo":4,"Siempre":5}
    EC = {"Muy en desacuerdo":1,"Moderadamente en desacuerdo":2,"Algo en desacuerdo":3,
          "Ni de acuerdo ni en desacuerdo":4,"Algo de acuerdo":5,
          "Moderadamente de acuerdo":6,"Muy de acuerdo":7}
    ED = {"Nunca":1,"Raramente":2,"Ocasionalmente":3,"Algunas veces":4,
          "Frecuentemente":5,"Casi siempre":6,"Siempre":7}

    DIMS = {
        "CT":(["CT1","CT2","CT3"],EA),"PT":(["PT1","PT2","PT3","PT4"],EA),
        "CL":(["CL1","CL2","CL3","CL4","CL5","CL6","CL7"],EA),
        "AC":(["AC1","AC2","AC3"],EA),"CR":(["CR1","CR2","CR3","CR4"],EA),
        "CoR":(["CoR1","CoR2","CoR3"],EA),"GC":(["GC1","GC2","GC3","GC4"],EA),
        "SM":(["SM1","SM2","SM3","SM4","SM5"],EA),
        "BU":(["BU1","BU2","BU3","BU4","BU5","BU6","BU7","BU8","BU9","BU10","BU11","BU12"],EB),
        "SAT":(["SAT1","SAT2","SAT3","SAT4","SAT5","SAT6","SAT7","SAT8","SAT9"],EC),
        "IR":(["IR1","IR2","IR3","IR4"],EC),"FT":(["FT1","FT2","FT3","FT4","FT5"],EC),
        "TF":(["TF1","TF2","TF3","TF4","TF5"],EC),
        "SOM":(["SOM1","SOM2","SOM3","SOM4","SOM5"],ED),
        "DL":(["DL1","DL2","DL3","DL4","DL5","DL6","DL7","DL8"],ED),
        "BP":(["BP1","BP2","BP3","BP4","BP5","BP6","BP7","BP8","BP9","BP10"],None),
    }
    RANGES = {"CT":(1,5),"PT":(1,5),"CL":(1,5),"AC":(1,5),"CR":(1,5),
              "CoR":(1,5),"GC":(1,5),"SM":(1,5),"BU":(1,5),
              "SAT":(1,7),"IR":(1,7),"FT":(1,7),"TF":(1,7),
              "SOM":(1,7),"DL":(1,7),"BP":(1,7)}
    LABELS = {
        "CT":"Control del Trabajo","PT":"Presion del Tiempo",
        "CL":"Compromiso del Lider","AC":"Apoyo de Companeros",
        "CR":"Claridad de Rol","CoR":"Conflicto de Rol",
        "GC":"Gestion del Cambio","SM":"Salud Mental Org.",
        "SAT":"Satisfaccion/Engagement","IR":"Intencion de Retiro",
        "FT":"Conflicto Familia-Trabajo","TF":"Conflicto Trabajo-Familia",
        "BU":"Burnout","BP":"Bienestar Percibido",
        "SOM":"Somatizacion","DL":"Desgaste Laboral",
    }
    d = df.copy()
    for dim,(cols,esc) in DIMS.items():
        for c in cols:
            d[c] = d[c].map(esc) if esc else pd.to_numeric(d[c], errors="coerce")
    d["IR1"] = 8 - d["IR1"]
    for dim,(cols,_) in DIMS.items():
        d[f"score_{dim}"] = d[cols].mean(axis=1)
    return d, DIMS, RANGES, LABELS

df_num, DIMS, RANGES, LABELS = load_data()
score_cols = [f"score_{d}" for d in DIMS]

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px 0;">
        <div style="font-size:20px; font-weight:900; color:#fff;">Bienestar Laboral</div>
        <div style="font-size:11px; color:{MUTED}; margin-top:4px;">EDA · Colombia · 2026</div>
    </div>
    <hr style="border-color:{BORDER}; margin:12px 0;">
    """, unsafe_allow_html=True)

    sel_sexo  = st.selectbox("Sexo",      ["Todos"] + sorted(df_num["Sexo"].unique().tolist()))
    sel_cargo = st.selectbox("Cargo",     ["Todos"] + sorted(df_num["Tipo_Cargo"].unique().tolist()))
    sel_sector= st.selectbox("Sector",    ["Todos"] + sorted(df_num["Sector"].unique().tolist()))
    sel_modal = st.selectbox("Modalidad", ["Todos"] + sorted(df_num["Modalidad"].unique().tolist()))

    st.markdown(f"<hr style='border-color:{BORDER};'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:10px; color:{MUTED}; text-align:center;'>400 trabajadores · 16 dimensiones psicosociales</div>", unsafe_allow_html=True)

# Filtros
df_f = df_num.copy()
if sel_sexo   != "Todos": df_f = df_f[df_f["Sexo"]       == sel_sexo]
if sel_cargo  != "Todos": df_f = df_f[df_f["Tipo_Cargo"] == sel_cargo]
if sel_sector != "Todos": df_f = df_f[df_f["Sector"]     == sel_sector]
if sel_modal  != "Todos": df_f = df_f[df_f["Modalidad"]  == sel_modal]

n_filt = len(df_f)
active = any(f != "Todos" for f in [sel_sexo, sel_cargo, sel_sector, sel_modal])

# ── Hero ───────────────────────────────────────────────────────────────────
pill = f"{n_filt} trabajadores con filtros activos" if active else f"Muestra completa · {n_filt} trabajadores"
st.markdown(f"""
<div class="hero">
    <div class="hero-title">Bienestar Laboral y Salud Mental Organizacional</div>
    <div class="hero-sub">Analisis Exploratorio de Datos &nbsp;·&nbsp; Colombia &nbsp;·&nbsp; 2026</div>
    <span class="hero-pill">{pill}</span>
</div>
""", unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────
bu_m  = df_f["score_BU"].mean()
dl_m  = df_f["score_DL"].mean()
sat_m = df_f["score_SAT"].mean()
ir_m  = df_f["score_IR"].mean()
cl_m  = df_f["score_CL"].mean()
bu_r  = (df_f["score_BU"] > 3).mean()

kpi_data = [
    (NEON["red"],    "BURNOUT",          f"{bu_m:.2f}/5",  f"{bu_r:.0%} en riesgo"),
    (NEON["orange"], "DESGASTE",         f"{dl_m:.2f}/7",  f"SD {df_f['score_DL'].std():.2f}"),
    (NEON["green"],  "SATISFACCION",     f"{sat_m:.2f}/7", f"SD {df_f['score_SAT'].std():.2f}"),
    (NEON["purple"], "INT. RETIRO",      f"{ir_m:.2f}/7",  f"{(df_f['score_IR']>4).mean():.0%} alto"),
    (NEON["cyan"],   "COMP. LIDER",      f"{cl_m:.2f}/5",  "factor protector"),
    (NEON["yellow"], "TRABAJADORES",     str(n_filt),      "en esta seleccion"),
]

cols_kpi = st.columns(6)
for col_k, (color, label, val, sub) in zip(cols_kpi, kpi_data):
    with col_k:
        st.markdown(f"""
        <div class="kpi" style="--kpi-color:{color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-val" style="color:{color} !important;">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Perfil  ",
    "  Ranking de Riesgo  ",
    "  Correlaciones  ",
    "  Comparaciones  ",
    "  Hallazgos  ",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PERFIL SOCIODEMOGRAFICO
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Distribucion por Sexo</div>
            <div class="chart-sub">Composicion de genero de la muestra</div>""",
            unsafe_allow_html=True)
        cnt = df_f["Sexo"].value_counts().reset_index()
        fig = px.pie(cnt, values="count", names="Sexo", hole=0.55,
                     color_discrete_sequence=[NEON["cyan"], NEON["pink"], NEON["purple"]])
        fig.update_traces(textfont=dict(size=13, color="white"),
                          marker=dict(line=dict(color=BG_CARD, width=3)))
        style_fig(fig, 300)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Tipo de Cargo</div>
            <div class="chart-sub">Distribucion por nivel jerarquico</div>""",
            unsafe_allow_html=True)
        cnt2 = df_f["Tipo_Cargo"].value_counts().reset_index()
        fig2 = px.bar(cnt2, x="count", y="Tipo_Cargo", orientation="h",
                      color="count", color_continuous_scale=["#1a1a3e", NEON["cyan"]],
                      text="count")
        fig2.update_traces(textfont=dict(size=12), marker_line_width=0,
                           textposition="outside", textfont_color="white")
        fig2.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Trabajadores")
        style_fig(fig2, 300)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Sector y Modalidad</div>
            <div class="chart-sub">Cruce entre sector y forma de trabajo</div>""",
            unsafe_allow_html=True)
        sm = df_f.groupby(["Sector","Modalidad"]).size().reset_index(name="n")
        fig3 = px.bar(sm, x="Sector", y="n", color="Modalidad", barmode="group",
                      color_discrete_sequence=[NEON["cyan"], NEON["purple"], NEON["pink"]])
        fig3.update_traces(marker_line_width=0)
        style_fig(fig3, 300)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Distribucion de Edad</div>
            <div class="chart-sub">Histograma con media de la muestra</div>""",
            unsafe_allow_html=True)
        fig4 = px.histogram(df_f, x="Edad", nbins=20,
                            color_discrete_sequence=[NEON["purple"]])
        fig4.update_traces(marker_line_width=0, opacity=0.85)
        fig4.add_vline(x=df_f["Edad"].mean(), line_dash="dash",
                       line_color=NEON["cyan"], line_width=2,
                       annotation_text=f"Media {df_f['Edad'].mean():.1f}",
                       annotation_font_color=NEON["cyan"])
        style_fig(fig4, 300)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Horas semana + experiencia
    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Horas Semanales de Trabajo</div>
            <div class="chart-sub">Distribucion de carga horaria reportada</div>""",
            unsafe_allow_html=True)
        fig5 = px.histogram(df_f, x="Horas_Semana", nbins=18,
                            color_discrete_sequence=[NEON["orange"]])
        fig5.update_traces(marker_line_width=0, opacity=0.85)
        fig5.add_vline(x=df_f["Horas_Semana"].mean(), line_dash="dash",
                       line_color=NEON["yellow"], line_width=2,
                       annotation_text=f"Media {df_f['Horas_Semana'].mean():.1f}h",
                       annotation_font_color=NEON["yellow"])
        style_fig(fig5, 300)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Nivel Educativo</div>
            <div class="chart-sub">Formacion academica de los participantes</div>""",
            unsafe_allow_html=True)
        edu = df_f["Nivel_Educativo"].value_counts().reset_index()
        fig6 = px.bar(edu, x="count", y="Nivel_Educativo", orientation="h",
                      color="count",
                      color_continuous_scale=["#1a0a2e", NEON["purple"]],
                      text="count")
        fig6.update_traces(textfont_color="white", textposition="outside", marker_line_width=0)
        fig6.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="N")
        style_fig(fig6, 300)
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — RANKING DE RIESGO
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    risk_rows = []
    for dim in DIMS:
        mn, mx = RANGES[dim]
        mean_r = df_f[f"score_{dim}"].mean()
        norm   = (mean_r - mn) / (mx - mn)
        pct    = (df_f[f"score_{dim}"] > (mx+mn)/2).mean() * 100
        risk_rows.append({"Codigo":dim,"Dimension":LABELS[dim],
                          "Media":round(mean_r,3),"Score":round(norm,3),
                          "PctAlto":round(pct,1)})
    risk_df = pd.DataFrame(risk_rows).sort_values("Score", ascending=False)

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Ranking de Riesgo Psicosocial</div>
            <div class="chart-sub">Score normalizado respecto al rango teorico de cada escala</div>""",
            unsafe_allow_html=True)
        bar_colors = [NEON["red"] if s >= 0.55 else
                      NEON["orange"] if s >= 0.40 else
                      NEON["green"] for s in risk_df["Score"]]
        fig_r = go.Figure(go.Bar(
            x=risk_df["Score"], y=risk_df["Dimension"],
            orientation="h",
            marker=dict(color=bar_colors,
                        line=dict(width=0)),
            text=[f"{v:.1%}" for v in risk_df["Score"]],
            textposition="outside",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1%}<extra></extra>",
        ))
        fig_r.add_vline(x=0.55, line_dash="dash", line_color=NEON["red"],
                        line_width=1.5, opacity=0.7)
        fig_r.add_vline(x=0.40, line_dash="dot",  line_color=NEON["orange"],
                        line_width=1.5, opacity=0.7)
        fig_r.update_layout(xaxis_range=[0, 1.12], yaxis={"categoryorder":"total ascending"})
        style_fig(fig_r, 500)
        st.plotly_chart(fig_r, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Radar de Riesgo</div>
            <div class="chart-sub">Perfil global de dimensiones criticas</div>""",
            unsafe_allow_html=True)
        dims_r = ["BU","DL","SOM","PT","CoR","IR","SAT","CL","SM","GC"]
        vals_r = [(df_f[f"score_{d}"].mean()-RANGES[d][0])/(RANGES[d][1]-RANGES[d][0])
                  for d in dims_r]
        labs_r = [LABELS[d] for d in dims_r]

        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(
            r=vals_r + [vals_r[0]], theta=labs_r + [labs_r[0]],
            fill="toself", name="Riesgo actual",
            line=dict(color=NEON["cyan"], width=2),
            fillcolor="rgba(0,212,255,0.12)",
        ))
        fig_rad.add_trace(go.Scatterpolar(
            r=[0.55]*len(labs_r)+[0.55], theta=labs_r+[labs_r[0]],
            mode="lines", name="Umbral alto",
            line=dict(color=NEON["red"], dash="dash", width=1.5),
        ))
        fig_rad.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER,
                                tickfont=dict(size=9, color=MUTED)),
                angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=9)),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        style_fig(fig_rad, 500)
        st.plotly_chart(fig_rad, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tabla resumen
    st.markdown(f"""<div class="chart-card">
        <div class="chart-title">Tabla Resumen de Riesgo por Dimension</div>
        <div class="chart-sub">Score normalizado y porcentaje de trabajadores en zona de riesgo</div>""",
        unsafe_allow_html=True)
    display_df = risk_df[["Dimension","Media","Score","PctAlto"]].copy()
    display_df.columns = ["Dimension","Media Raw","Score Riesgo","% Trabajadores en Riesgo"]
    display_df["Nivel"] = display_df["Score Riesgo"].apply(
        lambda x: "ALTO" if x >= 0.55 else "MODERADO" if x >= 0.40 else "BAJO")
    st.dataframe(display_df.style.background_gradient(
        subset=["Score Riesgo"], cmap="RdYlGn_r"),
        use_container_width=True, height=360)
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — CORRELACIONES
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    dims_all = list(DIMS.keys())
    n = len(dims_all)
    corr_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, _ = spearmanr(df_f[f"score_{dims_all[i]}"], df_f[f"score_{dims_all[j]}"])
            corr_mat[i, j] = round(r, 3)

    st.markdown(f"""<div class="chart-card">
        <div class="chart-title">Matriz de Correlacion de Spearman</div>
        <div class="chart-sub">Relaciones entre las 16 dimensiones psicosociales (rho)</div>""",
        unsafe_allow_html=True)
    fig_corr = px.imshow(corr_mat, x=dims_all, y=dims_all,
                         color_continuous_scale=["#ff4757","#1e293b","#00d4ff"],
                         zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
    fig_corr.update_traces(textfont=dict(size=9))
    style_fig(fig_corr, 560)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Relaciones Clave")
    pairs = [
        ("score_BU","score_DL","Burnout","Desgaste Laboral", NEON["red"]),
        ("score_BU","score_SOM","Burnout","Somatizacion",    NEON["pink"]),
        ("score_CL","score_SAT","Comp. Lider","Satisfaccion",NEON["cyan"]),
        ("score_SAT","score_IR","Satisfaccion","Int. Retiro",NEON["green"]),
    ]
    sc1, sc2 = st.columns(2)
    sc_cols = [sc1, sc2, sc1, sc2]

    for (xc, yc, xl, yl, col_n), sc_col in zip(pairs, sc_cols):
        r, p = spearmanr(df_f[xc], df_f[yc])
        x_v  = df_f[xc].values
        y_v  = df_f[yc].values
        m, b = np.polyfit(x_v, y_v, 1)
        xs   = np.linspace(x_v.min(), x_v.max(), 100)

        with sc_col:
            st.markdown(f"""<div class="chart-card">
                <div class="chart-title">{xl} vs {yl}</div>
                <div class="chart-sub">rho = {r:.3f} · p {'< 0.001' if p<0.001 else f'= {p:.3f}'}</div>""",
                unsafe_allow_html=True)
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=x_v, y=y_v, mode="markers",
                marker=dict(color=col_n, opacity=0.35, size=6,
                            line=dict(width=0)),
                name="Datos",
                hovertemplate=f"{xl}: %{{x:.2f}}<br>{yl}: %{{y:.2f}}<extra></extra>",
            ))
            fig_sc.add_trace(go.Scatter(
                x=xs, y=m*xs+b, mode="lines",
                line=dict(color="white", width=2, dash="dash"),
                name="Tendencia", showlegend=False,
            ))
            fig_sc.update_layout(xaxis_title=xl, yaxis_title=yl)
            style_fig(fig_sc, 280)
            st.plotly_chart(fig_sc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — COMPARACIONES
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        dim_sel = st.selectbox("Dimension", list(DIMS.keys()),
                               format_func=lambda d: LABELS[d],
                               index=list(DIMS.keys()).index("BU"))
    with cb:
        grp_sel = st.selectbox("Agrupar por",
                               ["Tipo_Cargo","Sector","Modalidad","Sexo",
                                "Nivel_Educativo","Tipo_Contrato"])

    score_s = f"score_{dim_sel}"
    st.markdown(f"""<div class="chart-card">
        <div class="chart-title">{LABELS[dim_sel]} por {grp_sel}</div>
        <div class="chart-sub">Violin plot con distribucion completa por grupo</div>""",
        unsafe_allow_html=True)
    fig_vio = px.violin(df_f, x=grp_sel, y=score_s, color=grp_sel,
                        box=True, points="outliers",
                        color_discrete_sequence=NEON_LIST,
                        labels={score_s: LABELS[dim_sel]})
    fig_vio.update_layout(showlegend=False)
    style_fig(fig_vio, 380)
    st.plotly_chart(fig_vio, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    cv1, cv2 = st.columns(2)
    with cv1:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Mapa de Calor por Cargo</div>
            <div class="chart-sub">Score promedio de cada dimension segun cargo</div>""",
            unsafe_allow_html=True)
        pivot = df_f.groupby("Tipo_Cargo")[score_cols].mean().round(2)
        pivot.columns = list(DIMS.keys())
        fig_h = px.imshow(pivot,
                          color_continuous_scale=["#0f172a", NEON["orange"], NEON["red"]],
                          text_auto=".2f", aspect="auto")
        fig_h.update_traces(textfont=dict(size=9))
        style_fig(fig_h, 320)
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cv2:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Burnout por Sector</div>
            <div class="chart-sub">Distribucion comparada entre sectores</div>""",
            unsafe_allow_html=True)
        fig_hist = px.histogram(df_f, x="score_BU", color="Sector",
                                nbins=22, barmode="overlay", opacity=0.7,
                                color_discrete_sequence=[NEON["cyan"], NEON["pink"]],
                                labels={"score_BU":"Burnout Score"})
        fig_hist.update_traces(marker_line_width=0)
        style_fig(fig_hist, 320)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Barras de medias por dimension y sector
    st.markdown(f"""<div class="chart-card">
        <div class="chart-title">Medias por Dimension y Sector</div>
        <div class="chart-sub">Comparacion de scores promedio normalizados entre sectores</div>""",
        unsafe_allow_html=True)
    means_sector = []
    for dim in DIMS:
        mn, mx = RANGES[dim]
        for sector in df_f["Sector"].unique():
            sub = df_f[df_f["Sector"]==sector][f"score_{dim}"]
            if len(sub) >= 5:
                norm = (sub.mean()-mn)/(mx-mn)
                means_sector.append({"Dimension":dim,"Sector":sector,"Score":round(norm,3)})
    ms_df = pd.DataFrame(means_sector)
    fig_ms = px.bar(ms_df, x="Dimension", y="Score", color="Sector",
                    barmode="group",
                    color_discrete_sequence=[NEON["cyan"], NEON["purple"], NEON["pink"]])
    fig_ms.update_traces(marker_line_width=0)
    fig_ms.add_hline(y=0.55, line_dash="dash", line_color=NEON["red"],
                     opacity=0.6, annotation_text="Umbral riesgo alto")
    style_fig(fig_ms, 360)
    st.plotly_chart(fig_ms, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — HALLAZGOS
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<br>", unsafe_allow_html=True)

    # Top 3 KPIs de riesgo
    top3 = risk_df.head(3)
    h1, h2, h3 = st.columns(3)
    for hcol, (_, row) in zip([h1, h2, h3], top3.iterrows()):
        color = NEON["red"] if row["Score"] >= 0.55 else NEON["orange"]
        with hcol:
            st.markdown(f"""
            <div class="kpi" style="--kpi-color:{color};">
                <div class="kpi-label">{row['Dimension']}</div>
                <div class="kpi-val" style="color:{color} !important;">{row['Score']:.1%}</div>
                <div class="kpi-sub">{row['PctAlto']:.1f}% sobre punto medio</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge de burnout
    hg1, hg2 = st.columns(2)
    with hg1:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Nivel de Burnout — Indicador</div>
            <div class="chart-sub">Media de la muestra seleccionada en escala 1-5</div>""",
            unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=bu_m,
            delta={"reference": 2.5, "valueformat": ".2f"},
            gauge={
                "axis": {"range":[1,5], "tickcolor": MUTED},
                "bar":  {"color": NEON["red"]},
                "bgcolor": BG_CARD2,
                "bordercolor": BORDER,
                "steps":[
                    {"range":[1,2.5], "color":"rgba(0,245,160,0.15)"},
                    {"range":[2.5,3.5],"color":"rgba(255,140,66,0.15)"},
                    {"range":[3.5,5],  "color":"rgba(255,71,87,0.15)"},
                ],
                "threshold":{"line":{"color":NEON["yellow"],"width":3},"value":3.5},
            },
            number={"font":{"color":"white","size":40},"suffix":"/5"},
        ))
        style_fig(fig_g, 280)
        st.plotly_chart(fig_g, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with hg2:
        st.markdown(f"""<div class="chart-card">
            <div class="chart-title">Satisfaccion vs Intencion de Retiro</div>
            <div class="chart-sub">Comparacion de medias en la muestra filtrada</div>""",
            unsafe_allow_html=True)
        comp_data = {
            "Dimension": ["Burnout","Desgaste","Somatizacion","Satisfaccion","Comp. Lider","Int. Retiro"],
            "Score Norm": [
                (bu_m-1)/4,
                (dl_m-1)/6,
                (df_f["score_SOM"].mean()-1)/6,
                (sat_m-1)/6,
                (cl_m-1)/4,
                (ir_m-1)/6,
            ],
        }
        comp_df = pd.DataFrame(comp_data)
        comp_colors = [NEON["red"],NEON["orange"],NEON["pink"],
                       NEON["green"],NEON["cyan"],NEON["purple"]]
        fig_comp = px.bar(comp_df, x="Score Norm", y="Dimension",
                          orientation="h",
                          color="Dimension",
                          color_discrete_sequence=comp_colors)
        fig_comp.update_traces(marker_line_width=0)
        fig_comp.update_layout(showlegend=False)
        fig_comp.add_vline(x=0.55, line_dash="dash", line_color=NEON["red"],
                           opacity=0.6, line_width=1.5)
        style_fig(fig_comp, 280)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Conclusiones dinamicas
    st.markdown("### Conclusiones Dinamicas")
    bu_lv = "ALTO" if bu_m >= 3.5 else "MODERADO" if bu_m >= 2.5 else "BAJO"
    bu_col= NEON["red"] if bu_m >= 3.5 else NEON["orange"] if bu_m >= 2.5 else NEON["green"]
    sat_lv= "ALTO" if sat_m >= 5 else "MODERADO" if sat_m >= 3.5 else "BAJO"
    sat_c = NEON["green"] if sat_m >= 5 else NEON["orange"] if sat_m >= 3.5 else NEON["red"]
    r_bd, p_bd = spearmanr(df_f["score_BU"], df_f["score_DL"])

    conclusiones = [
        (bu_col, "Burnout",
         f"Promedio {bu_m:.2f}/5 — nivel <b style='color:{bu_col}'>{bu_lv}</b>. "
         f"El {bu_r:.0%} de los trabajadores supera el umbral de riesgo."),
        (sat_c, "Satisfaccion y Retencion",
         f"Satisfaccion promedio {sat_m:.2f}/7 — nivel <b style='color:{sat_c}'>{sat_lv}</b>. "
         f"Intencion de retiro: {ir_m:.2f}/7."),
        (NEON["cyan"], "Liderazgo como Factor Protector",
         f"Compromiso del lider: {cl_m:.2f}/5. "
         f"El liderazgo actua como amortiguador del burnout en la muestra."),
        (NEON["pink"], "Correlacion Burnout y Desgaste",
         f"rho = {r_bd:.3f} — {'correlacion fuerte' if abs(r_bd)>=0.5 else 'correlacion moderada'}. "
         f"Ambas dimensiones se refuerzan mutuamente."),
    ]
    for col_n, titulo, texto in conclusiones:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER};
                    border-left:4px solid {col_n}; border-radius:12px;
                    padding:16px 20px; margin-bottom:12px;">
            <div style="font-size:11px; font-weight:700; color:{col_n};
                        text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">{titulo}</div>
            <div style="font-size:13px; color:{TEXT}; line-height:1.7;">{texto}</div>
        </div>
        """, unsafe_allow_html=True)

    # Recomendaciones
    st.markdown("### Recomendaciones Organizacionales")
    recs = [
        (NEON["red"],    "URGENTE",    "Intervencion en Burnout y Desgaste",
         "Programas de bienestar, redistribucion de carga laboral y acceso a salud mental ocupacional."),
        (NEON["red"],    "URGENTE",    "Gestion de Presion del Tiempo",
         "Revision de procesos, priorizacion de tareas y formacion en gestion del tiempo."),
        (NEON["orange"], "IMPORTANTE", "Fortalecimiento del Liderazgo",
         "Capacitacion en liderazgo transformacional y evaluaciones 360 grados."),
        (NEON["orange"], "IMPORTANTE", "Retencion del Talento",
         "Estrategias de reconocimiento, desarrollo profesional y clima positivo."),
        (NEON["green"],  "PREVENTIVA", "Conciliacion Trabajo-Familia",
         "Politicas de flexibilidad horaria para trabajadores con responsabilidades familiares."),
    ]
    for col_n, nivel, titulo, texto in recs:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER};
                    border-left:4px solid {col_n}; border-radius:12px;
                    padding:16px 20px; margin-bottom:10px;">
            <span style="font-size:9px; font-weight:700; color:{col_n};
                         text-transform:uppercase; letter-spacing:1.4px;">{nivel}</span>
            <div style="font-size:14px; font-weight:700; color:#fff; margin:5px 0 8px 0;">{titulo}</div>
            <div style="font-size:13px; color:{MUTED}; line-height:1.6;">{texto}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:11px; color:{MUTED}; text-align:center; margin-top:20px;'>"
                f"Hallazgos generados automaticamente — N={n_filt} trabajadores seleccionados</div>",
                unsafe_allow_html=True)
