"""
Dashboard Interactivo — Bienestar Laboral y Salud Mental Organizacional
Análisis Exploratorio de Datos · Colombia · 2026
Uso: streamlit run dashboard_bienestar_v3.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr, kruskal, mannwhitneyu, linregress

# ── Página ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bienestar Laboral · EDA Colombia",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

NEON = {
    "cyan":   "#00d4ff", "purple": "#a855f7", "pink":   "#ff6b9d",
    "green":  "#00f5a0", "orange": "#ff8c42", "yellow": "#ffd166",
    "blue":   "#4d9fff", "red":    "#ff4757",
}
NEON_LIST = list(NEON.values())
BG_DARK = "#0a0e1a"; BG_CARD = "#111827"; BG_CARD2 = "#0f172a"
BORDER  = "#1e293b";  TEXT   = "#e2e8f0"; MUTED   = "#64748b"

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif!important}}
.stApp{{background:{BG_DARK}!important}}
.main .block-container{{padding:16px 32px 32px 32px;max-width:100%}}
section[data-testid="stSidebar"]{{background:linear-gradient(180deg,#0d1225,#111827)!important;border-right:1px solid {BORDER}}}
section[data-testid="stSidebar"] *{{color:{TEXT}!important}}
.hero{{background:linear-gradient(135deg,#0d1225,#0f1f3d 45%,#1a0a2e);border:1px solid {BORDER};border-radius:18px;padding:28px 40px;margin-bottom:20px;position:relative;overflow:hidden}}
.hero-title{{font-size:26px;font-weight:900;background:linear-gradient(135deg,#ffffff,{NEON['cyan']});-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 6px 0}}
.hero-sub{{font-size:13px;color:#7ea8cc!important;margin:0}}
.hero-pill{{display:inline-block;margin-top:10px;background:rgba(0,212,255,0.1);color:{NEON['cyan']}!important;border:1px solid rgba(0,212,255,0.3);padding:4px 16px;border-radius:20px;font-size:11px;font-weight:600}}
.dynamic-alert{{background:linear-gradient(135deg,rgba(255,71,87,0.12),rgba(168,85,247,0.08));border:1px solid rgba(255,71,87,0.3);border-left:4px solid {NEON['red']};border-radius:12px;padding:16px 20px;margin:12px 0 20px 0}}
.dynamic-alert-title{{font-size:10px;font-weight:800;color:{NEON['red']}!important;text-transform:uppercase;letter-spacing:1.4px;margin-bottom:6px}}
.dynamic-alert-body{{font-size:13px;color:{TEXT}!important;line-height:1.75}}
.kpi{{background:{BG_CARD};border:1px solid {BORDER};border-radius:14px;padding:18px 14px;text-align:center;position:relative;overflow:hidden}}
.kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--kpi-color,{NEON['cyan']});box-shadow:0 0 12px var(--kpi-color,{NEON['cyan']})}}
.kpi-label{{font-size:9px;color:{MUTED}!important;font-weight:700;text-transform:uppercase;letter-spacing:1.4px;margin-bottom:6px}}
.kpi-val{{font-size:28px;font-weight:900;line-height:1;margin-bottom:4px}}
.kpi-sub{{font-size:11px;color:{MUTED}!important}}
.chart-card{{background:{BG_CARD};border:1px solid {BORDER};border-radius:16px;padding:20px 20px 12px 20px;margin-bottom:20px;box-shadow:0 4px 32px rgba(0,0,0,0.4)}}
.chart-title{{font-size:13px;font-weight:700;color:{TEXT}!important;margin:0 0 2px 0}}
.chart-sub{{font-size:11px;color:{MUTED}!important;margin:0 0 12px 0}}
.stTabs [data-baseweb="tab-list"]{{background:{BG_CARD};border-radius:12px;padding:4px;border:1px solid {BORDER};gap:4px}}
.stTabs [data-baseweb="tab"]{{border-radius:8px;font-weight:600;font-size:13px;color:{MUTED}!important;padding:8px 18px;background:transparent}}
.stTabs [aria-selected="true"]{{background:linear-gradient(135deg,rgba(0,212,255,0.15),rgba(168,85,247,0.15))!important;color:{NEON['cyan']}!important;border:1px solid rgba(0,212,255,0.25)!important}}
h1,h2,h3,p,.stMarkdown,.stMarkdown p{{color:{TEXT}!important}}
h3{{color:{NEON['cyan']}!important;font-size:15px!important}}
::-webkit-scrollbar{{width:5px}}::-webkit-scrollbar-track{{background:{BG_DARK}}}
::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:3px}}
hr{{border:none;height:1px;background:linear-gradient(90deg,transparent,{BORDER},transparent);margin:16px 0}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────
def style_fig(fig, height=380):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, family="Inter", size=12),
        title_text="",
        legend=dict(bgcolor="rgba(17,24,39,0.9)", bordercolor=BORDER,
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=16, r=16, t=16, b=16), height=height,
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=11))
    return fig


# ── Carga y codificación ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("bienestar_laboral_EDA.xlsx")
    EA  = {"Nunca":1,"Rara vez":2,"Algunas veces":3,"Frecuentemente":4,"Siempre":5}
    EB  = {"Nunca":1,"Raramente":2,"Algunas veces":3,"A menudo":4,"Siempre":5}
    EC  = {"Muy en desacuerdo":1,"Moderadamente en desacuerdo":2,"Algo en desacuerdo":3,
           "Ni de acuerdo ni en desacuerdo":4,"Algo de acuerdo":5,
           "Moderadamente de acuerdo":6,"Muy de acuerdo":7}
    ED  = {"Nunca":1,"Raramente":2,"Ocasionalmente":3,"Algunas veces":4,
           "Frecuentemente":5,"Casi siempre":6,"Siempre":7}
    DIMS = {
        "CT":(["CT1","CT2","CT3"],EA), "PT":(["PT1","PT2","PT3","PT4"],EA),
        "CL":(["CL1","CL2","CL3","CL4","CL5","CL6","CL7"],EA),
        "AC":(["AC1","AC2","AC3"],EA), "CR":(["CR1","CR2","CR3","CR4"],EA),
        "CoR":(["CoR1","CoR2","CoR3"],EA), "GC":(["GC1","GC2","GC3","GC4"],EA),
        "SM":(["SM1","SM2","SM3","SM4","SM5"],EA),
        "BU":(["BU1","BU2","BU3","BU4","BU5","BU6","BU7","BU8","BU9","BU10","BU11","BU12"],EB),
        "SAT":(["SAT1","SAT2","SAT3","SAT4","SAT5","SAT6","SAT7","SAT8","SAT9"],EC),
        "IR":(["IR1","IR2","IR3","IR4"],EC), "FT":(["FT1","FT2","FT3","FT4","FT5"],EC),
        "TF":(["TF1","TF2","TF3","TF4","TF5"],EC),
        "SOM":(["SOM1","SOM2","SOM3","SOM4","SOM5"],ED),
        "DL":(["DL1","DL2","DL3","DL4","DL5","DL6","DL7","DL8"],ED),
        "BP":(["BP1","BP2","BP3","BP4","BP5","BP6","BP7","BP8","BP9","BP10"],None),
    }
    RANGES = {
        "CT":(1,5),"PT":(1,5),"CL":(1,5),"AC":(1,5),"CR":(1,5),
        "CoR":(1,5),"GC":(1,5),"SM":(1,5),"BU":(1,5),
        "SAT":(1,7),"IR":(1,7),"FT":(1,7),"TF":(1,7),
        "SOM":(1,7),"DL":(1,7),"BP":(1,7),
    }
    TIPO = {
        "CT":"recurso","PT":"demanda","CL":"recurso","AC":"recurso",
        "CR":"recurso","CoR":"demanda","GC":"recurso","SM":"recurso",
        "BU":"demanda","SAT":"recurso","IR":"demanda","FT":"demanda",
        "TF":"demanda","SOM":"demanda","DL":"demanda","BP":"recurso",
    }
    LABELS = {
        "CT":"Control del Trabajo",    "PT":"Presión del Tiempo",
        "CL":"Compromiso del Líder",   "AC":"Apoyo de Compañeros",
        "CR":"Claridad de Rol",        "CoR":"Conflicto de Rol",
        "GC":"Gestión del Cambio",     "SM":"Salud Mental Org.",
        "SAT":"Satisfacción/Engagement","IR":"Intención de Retiro",
        "FT":"Conflicto Familia→Trabajo","TF":"Conflicto Trabajo→Familia",
        "BU":"Burnout / Agotamiento",  "BP":"Bienestar Percibido",
        "SOM":"Somatización",          "DL":"Desgaste Laboral",
    }
    d = df.copy()
    for dim,(cols,esc) in DIMS.items():
        for c in cols:
            d[c] = d[c].map(esc) if esc else pd.to_numeric(d[c], errors="coerce")
    d["IR1"] = 8 - d["IR1"]
    for dim,(cols,_) in DIMS.items():
        d[f"score_{dim}"] = d[cols].mean(axis=1)
    return d, DIMS, RANGES, LABELS, TIPO


df_num, DIMS, RANGES, LABELS, TIPO = load_data()
score_cols = [f"score_{d}" for d in DIMS]


def risk_index(series, dim):
    mn, mx = RANGES[dim]
    pct = (series.mean() - mn) / (mx - mn)
    return pct if TIPO[dim] == "demanda" else 1 - pct


def risk_table(df_sub):
    rows = []
    for dim in DIMS:
        mn, mx = RANGES[dim]
        media  = df_sub[f"score_{dim}"].mean()
        pct    = (media - mn) / (mx - mn)
        ridx   = pct if TIPO[dim] == "demanda" else 1 - pct
        nivel  = "ALTO" if ridx >= 0.55 else "MODERADO" if ridx >= 0.40 else "BAJO"
        rows.append({"Codigo":dim,"Dimension":LABELS[dim],"Tipo":TIPO[dim].capitalize(),
                     "Media":round(media,3),"Escala":f"1–{mx}",
                     "Idx Riesgo":round(ridx,3),"Nivel":nivel})
    return pd.DataFrame(rows).sort_values("Idx Riesgo", ascending=False).reset_index(drop=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 8px 0">
        <div style="font-size:20px;font-weight:900;color:#fff">Bienestar Laboral</div>
        <div style="font-size:11px;color:{MUTED};margin-top:4px">EDA · Colombia · 2026</div>
    </div><hr style="border-color:{BORDER};margin:10px 0">
    <div style="font-size:10px;color:{NEON['cyan']};font-weight:700;text-transform:uppercase;
                letter-spacing:1px;margin-bottom:8px">Filtros de población</div>
    """, unsafe_allow_html=True)

    sel_sexo   = st.selectbox("Sexo",      ["Todos"] + sorted(df_num["Sexo"].unique().tolist()))
    sel_cargo  = st.selectbox("Cargo",     ["Todos"] + sorted(df_num["Tipo_Cargo"].unique().tolist()))
    sel_sector = st.selectbox("Sector",    ["Todos"] + sorted(df_num["Sector"].unique().tolist()))
    sel_modal  = st.selectbox("Modalidad", ["Todos"] + sorted(df_num["Modalidad"].unique().tolist()))

    st.markdown(f"<hr style='border-color:{BORDER}'>"
                f"<div style='font-size:10px;color:{MUTED};text-align:center'>"
                "400 trabajadores · 16 dimensiones<br>Modelo JD-R · Spearman · K-Means</div>",
                unsafe_allow_html=True)

# ── Filtros ────────────────────────────────────────────────────────────────
df_f = df_num.copy()
if sel_sexo   != "Todos": df_f = df_f[df_f["Sexo"]       == sel_sexo]
if sel_cargo  != "Todos": df_f = df_f[df_f["Tipo_Cargo"] == sel_cargo]
if sel_sector != "Todos": df_f = df_f[df_f["Sector"]     == sel_sector]
if sel_modal  != "Todos": df_f = df_f[df_f["Modalidad"]  == sel_modal]

n_filt    = len(df_f)
active    = any(f != "Todos" for f in [sel_sexo, sel_cargo, sel_sector, sel_modal])
filtro_parts = []
if sel_sexo   != "Todos": filtro_parts.append(f"Sexo: {sel_sexo}")
if sel_cargo  != "Todos": filtro_parts.append(f"Cargo: {sel_cargo}")
if sel_sector != "Todos": filtro_parts.append(f"Sector: {sel_sector}")
if sel_modal  != "Todos": filtro_parts.append(f"Modalidad: {sel_modal}")
filtro_txt = " · ".join(filtro_parts) if filtro_parts else "Sin filtros — muestra completa"

# ── KPIs ───────────────────────────────────────────────────────────────────
bu_m  = df_f["score_BU"].mean()
dl_m  = df_f["score_DL"].mean()
sat_m = df_f["score_SAT"].mean()
ir_m  = df_f["score_IR"].mean()
cl_m  = df_f["score_CL"].mean()
bu_r  = (df_f["score_BU"] > 3).mean()

# ── Hero ───────────────────────────────────────────────────────────────────
pill = f"{n_filt} trabajadores · filtros activos" if active else f"Muestra completa · {n_filt} trabajadores"
st.markdown(f"""
<div class="hero">
    <div class="hero-title">Bienestar Laboral y Salud Mental Organizacional</div>
    <div class="hero-sub">Análisis Exploratorio de Datos &nbsp;·&nbsp; Colombia &nbsp;·&nbsp; 2026 &nbsp;·&nbsp; {filtro_txt}</div>
    <span class="hero-pill">{pill}</span>
</div>
""", unsafe_allow_html=True)

for col_k, (color, label, val, sub) in zip(st.columns(6), [
    (NEON["red"],    "BURNOUT",      f"{bu_m:.2f}/5",  f"{bu_r:.0%} en riesgo"),
    (NEON["orange"], "DESGASTE",     f"{dl_m:.2f}/7",  f"σ={df_f['score_DL'].std():.2f}"),
    (NEON["green"],  "SATISFACCIÓN", f"{sat_m:.2f}/7", f"σ={df_f['score_SAT'].std():.2f}"),
    (NEON["purple"], "INT. RETIRO",  f"{ir_m:.2f}/7",  f"{(df_f['score_IR']>4).mean():.0%} alto"),
    (NEON["cyan"],   "COMP. LÍDER",  f"{cl_m:.2f}/5",  "factor protector"),
    (NEON["yellow"], "TRABAJADORES", str(n_filt),       "en esta selección"),
]):
    with col_k:
        st.markdown(f"""<div class="kpi" style="--kpi-color:{color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-val" style="color:{color}!important">{val}</div>
            <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

# ── ALERTA DINÁMICA ────────────────────────────────────────────────────────
rt = risk_table(df_f)
top1, top2, top3 = rt.iloc[0], rt.iloc[1], rt.iloc[2]
urg_color = NEON["red"] if top1["Nivel"] == "ALTO" else NEON["orange"]
grupo_desc = f"para {filtro_txt}" if filtro_parts else "en la muestra completa"

st.markdown(f"""
<div class="dynamic-alert" style="border-left-color:{urg_color}">
    <div class="dynamic-alert-title">⚡ Análisis dinámico — {grupo_desc}</div>
    <div class="dynamic-alert-body">
        La dimensión de <b style="color:{urg_color}">mayor urgencia</b> es
        <b style="color:#fff">{top1['Dimension']}</b>
        (índice de riesgo: <b style="color:{urg_color}">{top1['Idx Riesgo']:.1%}</b> · {top1['Nivel']}).
        Le siguen <b style="color:#fff">{top2['Dimension']}</b> ({top2['Idx Riesgo']:.1%})
        y <b style="color:#fff">{top3['Dimension']}</b> ({top3['Idx Riesgo']:.1%}).
        &nbsp;|&nbsp; N = {n_filt} trabajadores analizados.
    </div>
</div>
<br>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Perfil  ", "  Ranking de Riesgo  ", "  Correlaciones  ",
    "  Comparaciones  ", "  Hallazgos  ",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PERFIL
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Distribución por Sexo</div>'
                    f'<div class="chart-sub">Composición de género de la muestra seleccionada</div>',
                    unsafe_allow_html=True)
        cnt = df_f["Sexo"].value_counts().reset_index()
        fig = px.pie(cnt, values="count", names="Sexo", hole=0.55,
                     color_discrete_sequence=[NEON["cyan"], NEON["pink"], NEON["purple"]])
        fig.update_traces(textfont=dict(size=13, color="white"),
                          marker=dict(line=dict(color=BG_CARD, width=3)))
        style_fig(fig, 290); st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Tipo de Cargo</div>'
                    f'<div class="chart-sub">Distribución por nivel jerárquico (n por categoría)</div>',
                    unsafe_allow_html=True)
        cnt2 = df_f["Tipo_Cargo"].value_counts().reset_index()
        fig2 = px.bar(cnt2, x="count", y="Tipo_Cargo", orientation="h",
                      color="count", color_continuous_scale=["#1a1a3e", NEON["cyan"]],
                      text="count", labels={"count":"Trabajadores","Tipo_Cargo":"Tipo de cargo"})
        fig2.update_traces(textfont=dict(size=12), marker_line_width=0,
                           textposition="outside", textfont_color="white")
        fig2.update_layout(coloraxis_showscale=False,
                           yaxis_title="Tipo de cargo", xaxis_title="Trabajadores")
        style_fig(fig2, 290); st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Sector y Modalidad</div>'
                    f'<div class="chart-sub">Cruce entre sector organizacional y forma de trabajo</div>',
                    unsafe_allow_html=True)
        sm = df_f.groupby(["Sector","Modalidad"]).size().reset_index(name="n")
        fig3 = px.bar(sm, x="Sector", y="n", color="Modalidad", barmode="group",
                      color_discrete_sequence=[NEON["cyan"], NEON["purple"], NEON["pink"]],
                      labels={"n":"Trabajadores"})
        fig3.update_traces(marker_line_width=0)
        fig3.update_layout(xaxis_title="Sector", yaxis_title="Trabajadores")
        style_fig(fig3, 290); st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Nivel Educativo</div>'
                    f'<div class="chart-sub">Formación académica de los participantes</div>',
                    unsafe_allow_html=True)
        edu = df_f["Nivel_Educativo"].value_counts().reset_index()
        fig6 = px.bar(edu, x="count", y="Nivel_Educativo", orientation="h",
                      color="count", color_continuous_scale=["#1a0a2e", NEON["purple"]],
                      text="count", labels={"count":"Trabajadores","Nivel_Educativo":"Nivel educativo"})
        fig6.update_traces(textfont_color="white", textposition="outside", marker_line_width=0)
        fig6.update_layout(coloraxis_showscale=False,
                           yaxis_title="Nivel educativo", xaxis_title="Trabajadores")
        style_fig(fig6, 290); st.plotly_chart(fig6, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Tipo de Contrato</div>'
                    f'<div class="chart-sub">Distribución por vinculación laboral</div>',
                    unsafe_allow_html=True)
        cont = df_f["Tipo_Contrato"].value_counts().reset_index()
        fig_c = px.bar(cont, x="count", y="Tipo_Contrato", orientation="h",
                       color="count", color_continuous_scale=["#0d2a1a", NEON["green"]],
                       text="count", labels={"count":"Trabajadores","Tipo_Contrato":"Contrato"})
        fig_c.update_traces(textfont_color="white", textposition="outside", marker_line_width=0)
        fig_c.update_layout(coloraxis_showscale=False,
                            yaxis_title="Tipo de contrato", xaxis_title="Trabajadores")
        style_fig(fig_c, 290); st.plotly_chart(fig_c, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c6:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Personas a Cargo</div>'
                    f'<div class="chart-sub">Distribución según responsabilidad de supervisión</div>',
                    unsafe_allow_html=True)
        pc = df_f["Personas_Cargo"].value_counts().reset_index()
        fig_pc = px.pie(pc, values="count", names="Personas_Cargo", hole=0.5,
                        color_discrete_sequence=[NEON["orange"], NEON["yellow"]],
                        labels={"Personas_Cargo":"Personas a cargo"})
        fig_pc.update_traces(textfont=dict(size=12, color="white"),
                             marker=dict(line=dict(color=BG_CARD, width=3)))
        style_fig(fig_pc, 290); st.plotly_chart(fig_pc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — RANKING DE RIESGO
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)
    risk_df = risk_table(df_f)
    c1, c2  = st.columns([3, 2])

    with c1:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Ranking de Riesgo Psicosocial</div>'
                    f'<div class="chart-sub">Índice normalizado 0–100 % · rojo ≥ 55 % · naranja 40–55 % · verde &lt; 40 %</div>',
                    unsafe_allow_html=True)
        bar_colors = [NEON["red"] if r>=0.55 else NEON["orange"] if r>=0.40 else NEON["green"]
                      for r in risk_df["Idx Riesgo"]]
        fig_r = go.Figure(go.Bar(
            x=risk_df["Idx Riesgo"], y=risk_df["Dimension"], orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.1%}  [{t}]" for v,t in zip(risk_df["Idx Riesgo"], risk_df["Tipo"])],
            textposition="outside", textfont=dict(color="white", size=10),
            customdata=np.stack([risk_df["Media"], risk_df["Escala"], risk_df["Nivel"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Índice: %{x:.1%}<br>Media: %{customdata[0]}<br>Nivel: %{customdata[2]}<extra></extra>",
        ))
        fig_r.add_vline(x=0.55, line_dash="dash", line_color=NEON["red"], line_width=1.5, opacity=0.7,
                        annotation_text="Riesgo alto", annotation_font_color=NEON["red"])
        fig_r.add_vline(x=0.40, line_dash="dot",  line_color=NEON["orange"], line_width=1.5, opacity=0.7,
                        annotation_text="Moderado",   annotation_font_color=NEON["orange"])
        fig_r.update_layout(xaxis_range=[0,1.18], yaxis={"categoryorder":"total ascending"},
                            xaxis_title="Índice de riesgo normalizado",
                            yaxis_title="Dimensión psicosocial")
        style_fig(fig_r, 520); st.plotly_chart(fig_r, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Radar de Riesgo</div>'
                    f'<div class="chart-sub">Perfil global · línea roja = umbral 55 %</div>',
                    unsafe_allow_html=True)
        dims_r = ["BU","DL","SOM","PT","CoR","IR","SAT","CL","SM","GC"]
        vals_r = [risk_index(df_f[f"score_{d}"], d) for d in dims_r]
        labs_r = [LABELS[d] for d in dims_r]
        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(r=vals_r+[vals_r[0]], theta=labs_r+[labs_r[0]],
            fill="toself", name="Riesgo actual",
            line=dict(color=NEON["cyan"], width=2), fillcolor="rgba(0,212,255,0.12)"))
        fig_rad.add_trace(go.Scatterpolar(r=[0.55]*len(labs_r)+[0.55], theta=labs_r+[labs_r[0]],
            mode="lines", name="Umbral 55%",
            line=dict(color=NEON["red"], dash="dash", width=1.5)))
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER,
                                       tickfont=dict(size=9, color=MUTED)),
                       angularaxis=dict(gridcolor=BORDER, tickfont=dict(size=9)),
                       bgcolor="rgba(0,0,0,0)"))
        style_fig(fig_rad, 520); st.plotly_chart(fig_rad, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f'<div class="chart-card"><div class="chart-title">Tabla Resumen — Todas las Dimensiones</div>'
                f'<div class="chart-sub">Índice de riesgo, media y nivel para el grupo seleccionado</div>',
                unsafe_allow_html=True)
    disp = risk_df[["Dimension","Tipo","Media","Escala","Idx Riesgo","Nivel"]].copy()
    disp.columns = ["Dimension","Tipo","Media","Escala","Índice Riesgo","Nivel"]
    st.dataframe(disp, use_container_width=True, height=360)
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

    st.markdown(f'<div class="chart-card"><div class="chart-title">Matriz de Correlación de Spearman — 16 dimensiones</div>'
                f'<div class="chart-sub">ρ Spearman · justificado por no-normalidad (Shapiro-Wilk p &lt; 0.05 en todas las dimensiones)</div>',
                unsafe_allow_html=True)
    fig_corr = px.imshow(corr_mat, x=dims_all, y=dims_all,
                         color_continuous_scale=["#ff4757","#1e293b","#00d4ff"],
                         zmin=-1, zmax=1, text_auto=".2f", aspect="auto",
                         labels={"x":"Dimension","y":"Dimension","color":"ρ"})
    fig_corr.update_traces(textfont=dict(size=9))
    fig_corr.update_layout(xaxis_title="Dimensión psicosocial", yaxis_title="Dimensión psicosocial")
    style_fig(fig_corr, 520); st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Relaciones clave — Scatter + Regresión lineal (con R²)")

    pairs = [
        ("score_BU",  "score_DL",  "Burnout (BU)",             "Desgaste Laboral (DL)",    NEON["red"]),
        ("score_BU",  "score_SOM", "Burnout (BU)",             "Somatización (SOM)",       NEON["pink"]),
        ("score_CL",  "score_SAT", "Compromiso del Líder (CL)","Satisfacción (SAT)",       NEON["cyan"]),
        ("score_SAT", "score_IR",  "Satisfacción (SAT)",       "Intención de Retiro (IR)", NEON["green"]),
    ]

    for (xc,yc,xl,yl,col_n), sc_col in zip(pairs, [st.columns(2)[0], st.columns(2)[1],
                                                     st.columns(2)[0], st.columns(2)[1]]):
        pass  # recalculated below

    sc1, sc2 = st.columns(2)
    sc_cols  = [sc1, sc2, sc1, sc2]

    for (xc,yc,xl,yl,col_n), sc_col in zip(pairs, sc_cols):
        r_sp, p_sp = spearmanr(df_f[xc], df_f[yc])
        x_v = df_f[xc].values; y_v = df_f[yc].values
        sl, ic, r_lr, p_lr, _ = linregress(x_v, y_v)
        r2  = r_lr ** 2
        xs  = np.linspace(x_v.min(), x_v.max(), 100)
        p_txt = "< 0.001" if p_sp < 0.001 else f"= {p_sp:.3f}"

        with sc_col:
            st.markdown(f'<div class="chart-card">'
                        f'<div class="chart-title">{xl} vs {yl}</div>'
                        f'<div class="chart-sub">ρ = {r_sp:.3f} (p {p_txt}) &nbsp;·&nbsp; '
                        f'R² = {r2:.3f} &nbsp;·&nbsp; ŷ = {sl:.2f}x + {ic:.2f}</div>',
                        unsafe_allow_html=True)
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(x=x_v, y=y_v, mode="markers",
                marker=dict(color=col_n, opacity=0.28, size=6, line=dict(width=0)),
                name="Datos",
                hovertemplate=f"{xl}: %{{x:.2f}}<br>{yl}: %{{y:.2f}}<extra></extra>"))
            fig_sc.add_trace(go.Scatter(x=xs, y=sl*xs+ic, mode="lines",
                line=dict(color="white", width=2.2, dash="dash"),
                name=f"R² = {r2:.3f}"))
            fig_sc.update_layout(xaxis_title=xl, yaxis_title=yl,
                                 legend=dict(x=0.02, y=0.98, font=dict(size=10)))
            style_fig(fig_sc, 280); st.plotly_chart(fig_sc, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — COMPARACIONES
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        dim_sel = st.selectbox("Dimensión a comparar", list(DIMS.keys()),
                               format_func=lambda d: LABELS[d],
                               index=list(DIMS.keys()).index("BU"))
    with cb:
        grp_sel = st.selectbox("Agrupar por",
                               ["Tipo_Cargo","Sector","Modalidad","Sexo",
                                "Nivel_Educativo","Tipo_Contrato"])

    score_s = f"score_{dim_sel}"
    grupos  = sorted(df_f[grp_sel].dropna().unique())
    grupos_data = [df_f[df_f[grp_sel]==g][score_s].dropna().values for g in grupos if
                   len(df_f[df_f[grp_sel]==g][score_s].dropna()) >= 3]

    if len(grupos_data) >= 2:
        if len(grupos_data) == 2:
            stat, p_val = mannwhitneyu(grupos_data[0], grupos_data[1], alternative="two-sided")
            test_name   = "Mann-Whitney U"
        else:
            stat, p_val = kruskal(*grupos_data)
            test_name   = "Kruskal-Wallis H"
        sig     = "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else "ns"))
        p_str   = "< 0.001" if p_val<0.001 else f"= {p_val:.4f}"
        sig_msg = "✓ Diferencia significativa." if p_val<0.05 else "Sin diferencias significativas (p > 0.05)."
        test_str = f"{test_name}: stat = {stat:.2f} · p {p_str} {sig} · {sig_msg}"
    else:
        test_str = "Insuficientes grupos para prueba estadística."

    # Strip plot — sin boxplot
    st.markdown(f'<div class="chart-card">'
                f'<div class="chart-title">{LABELS[dim_sel]} por {grp_sel.replace("_"," ")}</div>'
                f'<div class="chart-sub">{test_str}</div>',
                unsafe_allow_html=True)

    fig_strip = go.Figure()
    for i, g in enumerate(grupos):
        sub   = df_f[df_f[grp_sel]==g][score_s].dropna()
        color = NEON_LIST[i % len(NEON_LIST)]
        fig_strip.add_trace(go.Box(
            y=sub, name=str(g), marker_color=color,
            fillcolor="rgba(0,0,0,0)", line_color="rgba(0,0,0,0)",
            boxpoints="all", jitter=0.45, pointpos=0,
            marker=dict(size=5, opacity=0.4), showlegend=True,
        ))
        if len(sub) > 0:
            fig_strip.add_shape(type="line",
                x0=i-0.35, x1=i+0.35, y0=sub.mean(), y1=sub.mean(),
                line=dict(color=color, width=2.5))
            fig_strip.add_annotation(x=i, y=sub.mean(),
                text=f"μ={sub.mean():.2f}", showarrow=False, yshift=14,
                font=dict(size=10, color=color))

    fig_strip.update_layout(
        xaxis_title=grp_sel.replace("_"," "),
        yaxis_title=f"{LABELS[dim_sel]} [puntaje promedio]",
        showlegend=False,
    )
    style_fig(fig_strip, 400); st.plotly_chart(fig_strip, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    cv1, cv2 = st.columns(2)
    with cv1:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Mapa de Calor por Cargo</div>'
                    f'<div class="chart-sub">Score normalizado por dimensión y tipo de cargo</div>',
                    unsafe_allow_html=True)
        pivot = df_f.groupby("Tipo_Cargo")[score_cols].mean().round(2)
        pivot.columns = list(DIMS.keys())
        for dim in DIMS:
            mn, mx = RANGES[dim]; pivot[dim] = (pivot[dim]-mn)/(mx-mn)
        fig_h = px.imshow(pivot, color_continuous_scale=["#0f172a", NEON["orange"], NEON["red"]],
                          text_auto=".2f", aspect="auto",
                          labels={"x":"Dimension","y":"Tipo de cargo","color":"Score norm."})
        fig_h.update_traces(textfont=dict(size=9))
        fig_h.update_layout(xaxis_title="Dimensión psicosocial", yaxis_title="Tipo de cargo")
        style_fig(fig_h, 300); st.plotly_chart(fig_h, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with cv2:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Scores por Sector</div>'
                    f'<div class="chart-sub">Comparación de dimensiones normalizadas entre sectores</div>',
                    unsafe_allow_html=True)
        ms_rows = []
        for dim in DIMS:
            mn, mx = RANGES[dim]
            for sector in df_f["Sector"].unique():
                sub = df_f[df_f["Sector"]==sector][f"score_{dim}"]
                if len(sub) >= 5:
                    ms_rows.append({"Dimension":dim,"Sector":sector,
                                    "Score":round((sub.mean()-mn)/(mx-mn),3)})
        ms_df = pd.DataFrame(ms_rows)
        if ms_df.empty or ms_df["Sector"].nunique() < 2:
            st.markdown(f"<p style='color:{MUTED}; font-size:13px; padding:10px 0;'>No hay suficientes sectores para mostrar esta grafica con los filtros actuales.</p>", unsafe_allow_html=True)
        else:
            fig_ms = px.bar(ms_df, x="Dimension", y="Score", color="Sector", barmode="group",
                            color_discrete_sequence=[NEON["cyan"], NEON["purple"], NEON["pink"]],
                            labels={"Score":"Score normalizado"})
            fig_ms.update_traces(marker_line_width=0)
            fig_ms.add_hline(y=0.55, line_dash="dash", line_color=NEON["red"], opacity=0.6,
                             annotation_text="Umbral riesgo alto",
                             annotation_font_color=NEON["red"])
            fig_ms.update_layout(xaxis_title="Dimension psicosocial",
                                 yaxis_title="Score normalizado (0-1)")
            style_fig(fig_ms, 300); st.plotly_chart(fig_ms, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — HALLAZGOS
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<br>", unsafe_allow_html=True)

    for hcol, (_, row) in zip(st.columns(3), [(i, rt.iloc[i]) for i in range(3)]):
        color = NEON["red"] if row["Nivel"]=="ALTO" else NEON["orange"]
        with hcol:
            st.markdown(f"""<div class="kpi" style="--kpi-color:{color}">
                <div class="kpi-label">#{_+1} urgencia · {row['Tipo']}</div>
                <div class="kpi-val" style="color:{color}!important">{row['Idx Riesgo']:.1%}</div>
                <div class="kpi-sub">{row['Dimension']}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    hg1, hg2 = st.columns(2)

    with hg1:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Indicador de Burnout</div>'
                    f'<div class="chart-sub">Media del grupo seleccionado en escala 1–5 · umbral = 3.5</div>',
                    unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=bu_m,
            delta={"reference":2.5,"valueformat":".2f"},
            gauge={"axis":{"range":[1,5],"tickcolor":MUTED},
                   "bar":{"color":NEON["red"]},
                   "bgcolor":BG_CARD2,"bordercolor":BORDER,
                   "steps":[{"range":[1,2.5],"color":"rgba(0,245,160,0.15)"},
                             {"range":[2.5,3.5],"color":"rgba(255,140,66,0.15)"},
                             {"range":[3.5,5],"color":"rgba(255,71,87,0.15)"}],
                   "threshold":{"line":{"color":NEON["yellow"],"width":3},"value":3.5}},
            number={"font":{"color":"white","size":38},"suffix":"/5"},
        ))
        style_fig(fig_g, 280); st.plotly_chart(fig_g, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with hg2:
        st.markdown(f'<div class="chart-card"><div class="chart-title">Scores Clave Normalizados</div>'
                    f'<div class="chart-sub">Principales dimensiones en escala 0–1 para el grupo seleccionado</div>',
                    unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            "Dimension":["Burnout","Desgaste","Somatización","Satisfacción","Comp. Líder","Int. Retiro"],
            "Score":[(bu_m-1)/4,(dl_m-1)/6,(df_f["score_SOM"].mean()-1)/6,
                     (sat_m-1)/6,(cl_m-1)/4,(ir_m-1)/6],
        })
        fig_comp = px.bar(comp_df, x="Score", y="Dimension", orientation="h",
                          color="Dimension",
                          color_discrete_sequence=[NEON["red"],NEON["orange"],NEON["pink"],
                                                   NEON["green"],NEON["cyan"],NEON["purple"]],
                          labels={"Score":"Score normalizado (0–1)","Dimension":""})
        fig_comp.update_traces(marker_line_width=0)
        fig_comp.update_layout(showlegend=False,
                               xaxis_title="Score normalizado (0–1)", yaxis_title="Dimension")
        fig_comp.add_vline(x=0.55, line_dash="dash", line_color=NEON["red"], opacity=0.6,
                           annotation_text="Umbral alto", annotation_font_color=NEON["red"])
        style_fig(fig_comp, 280); st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Conclusiones dinámicas ─────────────────────────────────────────────
    st.markdown("### Conclusiones dinámicas del grupo seleccionado")

    bu_lv = "ALTO" if bu_m>=3.5 else "MODERADO" if bu_m>=2.5 else "BAJO"
    bu_c  = NEON["red"] if bu_m>=3.5 else NEON["orange"] if bu_m>=2.5 else NEON["green"]
    sat_lv= "ALTO" if sat_m>=5.0 else "MODERADO" if sat_m>=3.5 else "BAJO"
    sat_c = NEON["green"] if sat_m>=5.0 else NEON["orange"] if sat_m>=3.5 else NEON["red"]
    ir_lv = "ALTO" if ir_m>=4.5 else "MODERADO" if ir_m>=3.0 else "BAJO"
    ir_c  = NEON["red"] if ir_m>=4.5 else NEON["orange"] if ir_m>=3.0 else NEON["green"]

    r_bd, _ = spearmanr(df_f["score_BU"], df_f["score_DL"])
    r_si, _ = spearmanr(df_f["score_SAT"], df_f["score_IR"])
    sl_si, ic_si, r_lr_si, _, _ = linregress(df_f["score_SAT"], df_f["score_IR"])

    for col_n, titulo, texto in [
        (urg_color,
         f"Dimensión más urgente para este grupo",
         f"<b style='color:{urg_color}'>{top1['Dimension']}</b> presenta el mayor índice de riesgo "
         f"(<b>{top1['Idx Riesgo']:.1%}</b> · nivel {top1['Nivel']}) para los {n_filt} trabajadores "
         f"seleccionados ({filtro_txt}). Prioridad de intervención en este subgrupo."),
        (bu_c,
         "Burnout y agotamiento",
         f"Promedio {bu_m:.2f}/5 — nivel <b style='color:{bu_c}'>{bu_lv}</b>. "
         f"{bu_r:.0%} supera el umbral de riesgo (>3). "
         f"Correlación con Desgaste: ρ = {r_bd:.3f} ({'fuerte' if abs(r_bd)>=0.5 else 'moderada'})."),
        (sat_c,
         "Satisfacción y retención",
         f"Satisfacción {sat_m:.2f}/7 — <b style='color:{sat_c}'>{sat_lv}</b>. "
         f"Intención de retiro: {ir_m:.2f}/7 — <b style='color:{ir_c}'>{ir_lv}</b>. "
         f"SAT predice IR con ρ = {r_si:.3f} y R² = {r_si**2:.3f} "
         f"(explica el {r_si**2*100:.1f}% de la varianza). "
         f"Cada punto adicional en SAT reduce IR en {abs(sl_si):.2f} puntos."),
        (NEON["cyan"],
         "Liderazgo como factor protector",
         f"Compromiso del líder: {cl_m:.2f}/5. "
         f"El liderazgo actúa como amortiguador del burnout y promotor de satisfacción. "
         f"Mayor retorno de inversión de cualquier intervención organizacional."),
    ]:
        st.markdown(f"""
        <div style="background:{BG_CARD};border:1px solid {BORDER};
                    border-left:4px solid {col_n};border-radius:12px;
                    padding:16px 20px;margin-bottom:12px">
            <div style="font-size:11px;font-weight:700;color:{col_n};
                        text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">{titulo}</div>
            <div style="font-size:13px;color:{TEXT};line-height:1.75">{texto}</div>
        </div>""", unsafe_allow_html=True)

    # ── Recomendaciones ────────────────────────────────────────────────────
    st.markdown("### Recomendaciones organizacionales")

    for col_n, nivel, titulo, texto in [
        (NEON["red"],    "URGENTE",    "Intervención en Burnout y Desgaste",
         "Protocolo de detección temprana, redistribución de carga y acceso a salud mental. El riesgo es transversal a todos los cargos."),
        (NEON["red"],    "URGENTE",    "Gestión de Presión del Tiempo",
         "Revisión de procesos y priorización de tareas. El burnout no es exclusivo de ningún nivel jerárquico: es un problema estructural."),
        (NEON["orange"], "IMPORTANTE", "Fortalecimiento del Liderazgo",
         f"Capacitación en liderazgo transformacional. Mayor ROI: impacta simultáneamente SAT, BU e IR. CL→SAT con R² = 0.237."),
        (NEON["orange"], "IMPORTANTE", "Estrategia de Retención",
         f"SAT predice IR con R² = {r_si**2:.2f}. Implementar stay interviews semestrales y planes de carrera diferenciados."),
        (NEON["green"],  "PREVENTIVA", "Protocolo de Trabajo Híbrido",
         "Trabajadores híbridos reportan mayor desgaste. Establecer desconexión digital, equipamiento adecuado y cohesión de equipo."),
    ]:
        st.markdown(f"""
        <div style="background:{BG_CARD};border:1px solid {BORDER};
                    border-left:4px solid {col_n};border-radius:12px;
                    padding:16px 20px;margin-bottom:10px">
            <span style="font-size:9px;font-weight:700;color:{col_n};
                         text-transform:uppercase;letter-spacing:1.4px">{nivel}</span>
            <div style="font-size:14px;font-weight:700;color:#fff;margin:5px 0 6px 0">{titulo}</div>
            <div style="font-size:13px;color:{MUTED};line-height:1.6">{texto}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:11px;color:{MUTED};text-align:center;margin-top:20px'>"
                f"Análisis generado automáticamente · N = {n_filt} trabajadores · {filtro_txt}</div>",
                unsafe_allow_html=True)
