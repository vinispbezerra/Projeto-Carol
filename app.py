import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
import warnings
import logging
import re

# =====================================================
# CONFIGURAÇÕES INICIAIS
# =====================================================
st.set_page_config(page_title="Análise de Importação PMMA", layout="wide")

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

# =====================================================
# 🎨 IDENTIDADE VISUAL ARKEMA (cores reais do logo)
# =====================================================
ARKEMA_COLORS = {
    "primary":    "#45416A",   # Roxo/violeta escuro — "ARK"
    "secondary":  "#2E8B72",   # Verde-teal médio — intermediário do gradiente
    "accent":     "#70C0A7",   # Teal claro — "EMA"
    "gradient_start": "#45416A",
    "gradient_end":   "#70C0A7",
    "light":      "#F5F5F8",
    "white":      "#FFFFFF",
    "gray":       "#6B7280",
    "dark_gray":  "#374151",
    "mid_purple": "#5A5490",   # Tom intermediário do gradiente do logo
}

ARKEMA_PALETTE = [
    "#45416A",   # Roxo Arkema
    "#70C0A7",   # Teal Arkema
    "#2E8B72",   # Verde médio
    "#8C87B8",   # Roxo claro
    "#A8D8CB",   # Teal suave
    "#2D6A8F",   # Azul complementar
    "#9B8EC4",   # Lavanda
    "#3DAF90",   # Verde-água vibrante
    "#6B5FA0",   # Roxo médio
    "#52B5A0",   # Teal intermediário
    "#B8B3D8",   # Lilás claro
    "#1E7A60",   # Verde escuro
]

st.markdown(f"""
<style>
    /* Fundo principal */
    .stApp {{
        background-color: #F5F5F8;
    }}
    /* Sidebar com gradiente das cores do logo */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #45416A 0%, #2E8B72 60%, #70C0A7 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: {ARKEMA_COLORS['white']} !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stRadio label {{
        color: #D6EEE8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    /* Títulos */
    h1, h2, h3 {{
        color: {ARKEMA_COLORS['primary']};
        font-family: 'Georgia', serif;
    }}
    /* Botões e highlights */
    .stButton > button {{
        background: linear-gradient(90deg, #45416A, #70C0A7);
        color: white;
        border: none;
        border-radius: 4px;
    }}
    /* Métricas */
    [data-testid="metric-container"] {{
        background: white;
        border-left: 4px solid {ARKEMA_COLORS['primary']};
        border-radius: 4px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    /* Header bar */
    .arkema-header {{
        background: linear-gradient(135deg, #45416A 0%, #2E8B72 60%, #70C0A7 100%);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }}
    .arkema-header h1 {{
        color: white !important;
        margin: 0;
        font-size: 1.6rem;
    }}
    .arkema-header span {{
        color: #D6EEE8;
        font-size: 2rem;
    }}
    .arkema-red-bar {{
        height: 4px;
        background: linear-gradient(90deg, #45416A, #70C0A7);
        border-radius: 2px;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown(f"""
<div class="arkema-header">
    <span>📦</span>
    <div>
        <h1>Análise de Importação — PMMA</h1>
        <p style="color:#94A3B8; margin:0; font-size:0.85rem;">Inteligência Comercial · Arkema</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MAPEAMENTO NCM → NOME DO PRODUTO
# =====================================================
NCM_NOMES = {
    "39061000": "39061000 – PMMA (Polimetilmetacrilato)",
    "39069090": "39069090 – Outros polímeros acrílicos",
    "39069010": "39069010 – Polímeros acrílicos em formas primárias",
    "39064000": "39064000 – Copolímeros de acrilonitrila",
    "32091000": "32091000 – Tintas e vernizes acrílicos",
}

def get_ncm_label(ncm_val):
    ncm_str = str(ncm_val).strip()
    return NCM_NOMES.get(ncm_str, ncm_str)

# =====================================================
# AGRUPAMENTO DE EXPORTADORES
# =====================================================
EXPORT_GROUPS = {
    r"(?i)\bLX\b|ROHM|LUCITE|ALTUGLAS": "LX / Röhm / Lucite",
    r"(?i)\bMITSUBISHI\b":              "Mitsubishi",
    r"(?i)\bSUMITOMO\b":                "Sumitomo",
    r"(?i)\bCHIMEI\b":                  "Chi Mei",
    r"(?i)\bARKEMA\b":                  "Arkema",
    r"(?i)\bEVONIK\b":                  "Evonik",
    r"(?i)\bINEOS\b":                   "Ineos",
    r"(?i)\bDOW\b|\bDOWDUPONT\b":       "Dow",
    r"(?i)\bBASF\b":                    "BASF",
    r"(?i)\bSABIC\b":                   "SABIC",
}

def agrupar_exportador(nome):
    if pd.isna(nome):
        return "Outros"
    for pattern, grupo in EXPORT_GROUPS.items():
        if re.search(pattern, str(nome)):
            return grupo
    return str(nome)[:40]  # Trunca nomes muito longos

# =====================================================
# HELPERS DE GRÁFICO
# =====================================================
def apply_arkema_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=ARKEMA_COLORS["secondary"]), x=0.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=11, color=ARKEMA_COLORS["dark_gray"]),
        height=height,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(size=9),
        ),
        colorway=ARKEMA_PALETTE,
        xaxis=dict(showgrid=True, gridcolor="#F0F0F0", linecolor="#E5E7EB"),
        yaxis=dict(showgrid=True, gridcolor="#F0F0F0", linecolor="#E5E7EB"),
    )
    return fig

def add_mean_line(fig, df, col_x, col_y, label="Média"):
    media = df[col_y].mean()
    fig.add_hline(
        y=media,
        line_dash="dash",
        line_color="#70C0A7",
        line_width=1.5,
        annotation_text=f"{label}: {media:,.2f}",
        annotation_position="top right",
        annotation_font=dict(color="#70C0A7", size=10),
    )
    return fig

# =====================================================
# UPLOAD DO ARQUIVO
# =====================================================
uploaded_file = st.file_uploader(
    "Carregue o arquivo de dados (Excel ou CSV)",
    type=["xlsx", "csv"],
    help="Formatos aceitos: .xlsx ou .csv"
)

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    df = pd.DataFrame()

    try:
        if file_extension == "xlsx":
            try:
                df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
            except ValueError:
                st.warning("Aba 'Sheet1' não encontrada. Lendo a primeira aba.")
                df = pd.read_excel(uploaded_file, sheet_name=0)
        elif file_extension == "csv":
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    # --- Normalização de colunas ---
    df.columns = [col.strip() for col in df.columns]

    column_mapping = {
        "Peso líquido": "Peso",
        "VALOR FOB ESTIMADO TOTAL": "Valor_FOB",
        "VALOR CIF TOTAL": "Valor_CIF",
        "QTD Estatística": "Qtd_Estatística",
        "Qtd. de operações estimada": "Qtd_Estatística",
        "Descrição produto": "Descrição",
        "PAIS DE ORIGEM": "País",
        "PAÍS DE ORIGEM": "País",
        "País de aquisição": "País_Aquisição",
        "URF de Entrada": "URF_Entrada",
        "PROVÁVEL IMPORTADOR": "Importador",
        "PROVÁVEL EXPORTADOR": "Exportador",
        "NCM's": "NCM",
        "NCM": "NCM",
        "MODAL": "Modal",
        "Incoterm": "Incoterm",
        "Valor CIF Unitário": "CIF_Unitário",
        "CIF Unitário": "CIF_Unitário",
        "Valor FOB Estimado Unitário": "FOB_Unitário",
    }

    renamed_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
    if "QTD Estatística" in df.columns and "Qtd. de operações estimada" in df.columns:
        df.drop(columns=["QTD Estatística"], inplace=True)
        if "QTD Estatística" in renamed_cols:
            del renamed_cols["QTD Estatística"]

    df = df.rename(columns=renamed_cols)
    if not df.empty:
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # --- Tratamento de data ---
    def safe_to_datetime(dt_val):
        if pd.isna(dt_val): return pd.NaT
        dt_str = str(dt_val).strip()
        if '-' in dt_str or '/' in dt_str:
            return pd.to_datetime(dt_str, errors='coerce')
        try:
            if '.' in dt_str: dt_str = dt_str.split('.')[0]
            if dt_str.isdigit() and len(dt_str) >= 6:
                return pd.to_datetime(dt_str[:6], format="%Y%m", errors='coerce')
        except:
            pass
        return pd.NaT

    if "ANO/MÊS" in df.columns:
        df["ANO/MÊS"] = df["ANO/MÊS"].apply(safe_to_datetime)
        df.dropna(subset=["ANO/MÊS"], inplace=True)
    else:
        st.error("Coluna 'ANO/MÊS' não encontrada.")
        st.stop()

    # --- Limpeza numérica (robusto: trata floats, strings e NaN) ---
    def clean_currency_string(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip().replace(" ", "")
        if not val or val.lower() in ("nan", "none", "-"):
            return np.nan
        if "," in val and "." in val:
            return float(val.replace(".", "").replace(",", "."))
        elif "," in val:
            return float(val.replace(",", "."))
        try:
            return float(val)
        except ValueError:
            return np.nan

    numeric_cols = ["Peso", "Valor_FOB", "Valor_CIF", "Qtd_Estatística", "CIF_Unitário", "FOB_Unitário"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_currency_string)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # --- Tratamento de NCM ---
    if "NCM" in df.columns:
        df["NCM"] = df["NCM"].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.', '', 1).isdigit() else str(x)
        )
        df["NCM_Label"] = df["NCM"].apply(get_ncm_label)

    # ✅ MELHORIA 1: CIF Unitário mais realista (Valor_CIF / Peso)
    if "Valor_CIF" in df.columns and "Peso" in df.columns:
        df["CIF_Unitário_Calc"] = df.apply(
            lambda r: r["Valor_CIF"] / r["Peso"] if r["Peso"] > 0 else np.nan, axis=1
        )
        # Prefere coluna calculada, mantém original como fallback
        if "CIF_Unitário" not in df.columns or df["CIF_Unitário"].sum() == 0:
            df["CIF_Unitário"] = df["CIF_Unitário_Calc"]
        else:
            df["CIF_Unitário"] = df["CIF_Unitário_Calc"]

    # Peso em toneladas
    df["Peso_Ton"] = df["Peso"] / 1000 if "Peso" in df.columns else 0

    # ✅ MELHORIA 4: Agrupamento de exportadores
    if "Exportador" in df.columns:
        df["Exportador_Grupo"] = df["Exportador"].apply(agrupar_exportador)

    # =====================================================
    # SIDEBAR – FILTROS
    # =====================================================
    st.sidebar.markdown(
        '<div style="padding:1rem 0; border-bottom:1px solid rgba(255,255,255,0.15); margin-bottom:1rem;">'
        '<p style="font-size:1.2rem; font-weight:700; letter-spacing:0.15em; '
        'background:linear-gradient(90deg,#B8B3D8,#A8D8CB); -webkit-background-clip:text; '
        '-webkit-text-fill-color:transparent; margin:0;">ARKEMA</p>'
        '<p style="font-size:0.7rem; color:#D6EEE8; margin:0;">Painel de Importação</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.header("🔍 Filtros")

    # ✅ MELHORIA 5: NCM com nome do produto
    if "NCM_Label" in df.columns:
        ncm_labels = sorted(df["NCM_Label"].dropna().unique().tolist())
        ncm_default = [n for n in ncm_labels if "39061000" in n]
        sel_ncm_labels = st.sidebar.multiselect("Filtrar por NCM:", options=ncm_labels, default=ncm_default)
        # Reverter para código NCM para filtrar
        sel_ncm = [lbl.split(" – ")[0].strip() if " – " in lbl else lbl for lbl in sel_ncm_labels]
    else:
        ncm_list = sorted(df["NCM"].dropna().unique().tolist()) if "NCM" in df.columns else []
        ncm_default = [n for n in ncm_list if "39061000" in n]
        sel_ncm = st.sidebar.multiselect("Filtrar por NCM:", options=ncm_list, default=ncm_default)

    df_filtrado = df.copy()
    if sel_ncm:
        df_filtrado = df_filtrado[df_filtrado["NCM"].isin(sel_ncm)]

    # ✅ MELHORIA 6: Descrição com palavras-chave e Select All
    if "Descrição" in df_filtrado.columns:
        kw_input = st.sidebar.text_input("🔎 Palavra-chave (Descrição):", placeholder="ex: granulado, folha...")
        descricoes_list = sorted(df_filtrado["Descrição"].dropna().unique().tolist())
        if kw_input:
            descricoes_list = [d for d in descricoes_list if kw_input.lower() in d.lower()]

        select_all_desc = st.sidebar.checkbox("Selecionar todas as descrições", value=False)
        sel_descricoes = st.sidebar.multiselect(
            "Filtrar por Descrição:",
            options=descricoes_list,
            default=descricoes_list if select_all_desc else []
        )
        if sel_descricoes:
            df_filtrado = df_filtrado[df_filtrado["Descrição"].isin(sel_descricoes)]

    importadores_list = sorted(df_filtrado["Importador"].dropna().unique().tolist()) if "Importador" in df_filtrado.columns else []
    sel_importadores = st.sidebar.multiselect("Pesquisar Importadores:", options=importadores_list)
    if sel_importadores:
        df_filtrado = df_filtrado[df_filtrado["Importador"].isin(sel_importadores)]

    exportadores_list = sorted(df_filtrado["Exportador"].dropna().unique().tolist()) if "Exportador" in df_filtrado.columns else []
    sel_exportadores = st.sidebar.multiselect("Pesquisar Exportadores:", options=exportadores_list)
    if sel_exportadores:
        df_filtrado = df_filtrado[df_filtrado["Exportador"].isin(sel_exportadores)]

    st.sidebar.markdown("---")
    menu = st.sidebar.radio("Navegação:", ["Análise Histórica", "Análise por Empresa", "Previsão"])

    # =====================================================
    # 📈 ANÁLISE HISTÓRICA
    # =====================================================
    if menu == "Análise Histórica":
        st.markdown('<div class="arkema-red-bar"></div>', unsafe_allow_html=True)
        st.subheader("📈 Painel de Análise Histórica")

        if df_filtrado.empty:
            st.warning("Nenhum dado encontrado para os filtros selecionados.")
        else:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                pais_col = "País" if "País" in df_filtrado.columns else "País_Aquisição"
                paises_options = sorted(df_filtrado[pais_col].dropna().unique().tolist()) if pais_col in df_filtrado.columns else []
                sel_paises = st.multiselect("Filtrar por País de Origem:", paises_options)
                if sel_paises:
                    df_filtrado = df_filtrado[df_filtrado[pais_col].isin(sel_paises)]
            with col_f2:
                group_opts = ["Nenhum"] + [c for c in ["Descrição", "País", "Importador", "Exportador_Grupo", "Modal", "Incoterm", "NCM_Label"] if c in df_filtrado.columns]
                group_by_col = st.selectbox("Agrupar evolução temporal por:", group_opts)

            group_cols = ["ANO/MÊS"]
            if group_by_col != "Nenhum":
                group_cols.append(group_by_col)

            df_grouped = df_filtrado.groupby(group_cols).agg(
                Peso=('Peso', 'sum'),
                Valor_CIF=('Valor_CIF', 'sum'),
                Peso_Ton=('Peso_Ton', 'sum'),
            ).reset_index()

            df_grouped['CIF_Unitário'] = df_grouped.apply(
                lambda row: row['Valor_CIF'] / row['Peso'] if row['Peso'] > 0 else 0, axis=1
            )

            color_col = group_by_col if group_by_col != "Nenhum" else None

            # --- Gráfico Peso ---
            fig_peso = px.line(
                df_grouped, x="ANO/MÊS", y="Peso_Ton",
                color=color_col,
                title="Evolução de Volume (Toneladas)",
                markers=True,
                color_discrete_sequence=ARKEMA_PALETTE,
                text="Peso_Ton",
            )
            # ✅ MELHORIA 8: Rótulos de dados
            fig_peso.update_traces(
                texttemplate="%{text:.1f}",
                textposition="top center",
                textfont=dict(size=8),
            )
            fig_peso = apply_arkema_layout(fig_peso, height=480)
            fig_peso.update_yaxes(title_text="Volume (ton)")
            # ✅ MELHORIA 2: Linha de média
            media_ton = df_grouped['Peso_Ton'].mean()
            fig_peso.add_hline(
                y=media_ton, line_dash="dash", line_color="#70C0A7", line_width=1.5,
                annotation_text=f"Média: {media_ton:,.1f} ton",
                annotation_position="top right",
                annotation_font=dict(color="#70C0A7", size=10),
            )
            st.plotly_chart(fig_peso, use_container_width=True)

            # --- Gráfico CIF Unitário ---
            fig_cif_u = px.line(
                df_grouped, x="ANO/MÊS", y="CIF_Unitário",
                color=color_col,
                title="Evolução CIF Unitário (US$/kg)",
                markers=True,
                color_discrete_sequence=ARKEMA_PALETTE,
                text="CIF_Unitário",
            )
            fig_cif_u.update_traces(
                texttemplate="US$ %{text:.4f}",
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate="Data: %{x}<br>CIF Unitário: US$ %{y:.4f}/kg",
            )
            fig_cif_u = apply_arkema_layout(fig_cif_u, height=480)
            fig_cif_u.update_yaxes(title_text="US$/kg")
            media_cif = df_grouped['CIF_Unitário'].mean()
            fig_cif_u.add_hline(
                y=media_cif, line_dash="dash", line_color="#70C0A7", line_width=1.5,
                annotation_text=f"Média: US$ {media_cif:,.4f}/kg",
                annotation_position="top right",
                annotation_font=dict(color="#70C0A7", size=10),
            )
            st.plotly_chart(fig_cif_u, use_container_width=True)

            # =====================================================
            # ✅ MELHORIA 2: NOVOS GRÁFICOS
            # =====================================================
            st.markdown("---")
            st.subheader("🏢 Análise por Empresa")

            col_new1, col_new2 = st.columns(2)

            # Gráfico: Volume (ton) × Importador (Company)
            with col_new1:
                if "Importador" in df_filtrado.columns:
                    df_vol_empresa = (
                        df_filtrado.groupby("Importador")["Peso_Ton"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(15)
                        .reset_index()
                    )
                    fig_vol_emp = px.bar(
                        df_vol_empresa, x="Importador", y="Peso_Ton",
                        title="Volume (Toneladas) × Importador",
                        color="Importador",
                        color_discrete_sequence=ARKEMA_PALETTE,
                        text="Peso_Ton",
                    )
                    fig_vol_emp.update_traces(
                        texttemplate="%{text:.1f} ton",
                        textposition="outside",
                        textfont=dict(size=9),
                    )
                    fig_vol_emp = apply_arkema_layout(fig_vol_emp, height=460)
                    fig_vol_emp.update_xaxes(tickangle=-40, title_text="")
                    fig_vol_emp.update_yaxes(title_text="Volume (ton)")
                    fig_vol_emp.update_layout(showlegend=False)
                    st.plotly_chart(fig_vol_emp, use_container_width=True)

            # Gráfico: Exportador_Grupo × CIF Unitário médio (Price)
            with col_new2:
                if "Exportador_Grupo" in df_filtrado.columns:
                    df_exp_price = (
                        df_filtrado.groupby("Exportador_Grupo")
                        .apply(lambda x: x["Valor_CIF"].sum() / x["Peso"].sum() if x["Peso"].sum() > 0 else 0)
                        .reset_index(name="CIF_Medio")
                        .sort_values("CIF_Medio", ascending=False)
                    )
                    fig_exp_price = px.bar(
                        df_exp_price, x="Exportador_Grupo", y="CIF_Medio",
                        title="Exportador × Preço CIF Médio (US$/kg)",
                        color="CIF_Medio",
                        color_continuous_scale=["#45416A", "#70C0A7"],
                        text="CIF_Medio",
                    )
                    fig_exp_price.update_traces(
                        texttemplate="US$ %{text:.4f}",
                        textposition="outside",
                        textfont=dict(size=9),
                    )
                    fig_exp_price = apply_arkema_layout(fig_exp_price, height=460)
                    fig_exp_price.update_xaxes(tickangle=-40, title_text="")
                    fig_exp_price.update_yaxes(title_text="CIF Médio (US$/kg)")
                    fig_exp_price.update_layout(coloraxis_showscale=False, showlegend=False)
                    st.plotly_chart(fig_exp_price, use_container_width=True)

            # =====================================================
            # DETALHAMENTO
            # =====================================================
            st.markdown("---")
            st.subheader("🔎 Detalhamento dos Dados")
            cols_show = ["ANO/MÊS", "NCM_Label", "Descrição", "País", "Peso_Ton", "CIF_Unitário", "Valor_CIF", "Importador", "Exportador_Grupo"]
            cols_available = [c for c in cols_show if c in df_filtrado.columns]
            df_display = df_filtrado[cols_available].sort_values("ANO/MÊS", ascending=False)

            fmt = {}
            if "CIF_Unitário" in df_display.columns:
                fmt["CIF_Unitário"] = "US$ {:,.4f}"
            if "Valor_CIF" in df_display.columns:
                fmt["Valor_CIF"] = "US$ {:,.2f}"
            if "Peso_Ton" in df_display.columns:
                fmt["Peso_Ton"] = "{:,.3f} ton"

            st.dataframe(
                df_display.style.format(fmt, decimal=',', thousands='.'),
                use_container_width=True,
                height=400,
            )

    # =====================================================
    # 🏢 ANÁLISE POR EMPRESA (menu dedicado)
    # =====================================================
    elif menu == "Análise por Empresa":
        st.markdown('<div class="arkema-red-bar"></div>', unsafe_allow_html=True)
        st.subheader("🏢 Análise por Empresa e Exportador")

        if df_filtrado.empty:
            st.warning("Nenhum dado encontrado para os filtros selecionados.")
        else:
            tab1, tab2, tab3 = st.tabs(["📦 Volume por Importador", "💲 Preço por Exportador", "🌍 Mapa de Origens"])

            with tab1:
                top_n = st.slider("Top N importadores:", 5, 30, 10)
                df_top_imp = (
                    df_filtrado.groupby("Importador")["Peso_Ton"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .reset_index()
                ) if "Importador" in df_filtrado.columns else pd.DataFrame()

                if not df_top_imp.empty:
                    fig_imp = px.bar(
                        df_top_imp, x="Peso_Ton", y="Importador",
                        orientation='h',
                        title=f"Top {top_n} Importadores por Volume",
                        color="Peso_Ton",
                        color_continuous_scale=["#45416A", "#70C0A7"],
                        text="Peso_Ton",
                    )
                    fig_imp.update_traces(
                        texttemplate="%{text:.1f} ton",
                        textposition="outside",
                        textfont=dict(size=10),
                    )
                    fig_imp = apply_arkema_layout(fig_imp, height=max(400, top_n * 28))
                    fig_imp.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_imp, use_container_width=True)

            with tab2:
                if "Exportador_Grupo" in df_filtrado.columns:
                    df_exp_grp = (
                        df_filtrado.groupby("Exportador_Grupo")
                        .agg(
                            CIF_Medio=('CIF_Unitário', 'mean'),
                            Volume_Ton=('Peso_Ton', 'sum'),
                            Operacoes=('ANO/MÊS', 'count'),
                        )
                        .reset_index()
                        .sort_values("CIF_Medio", ascending=False)
                    )

                    fig_bubble = px.scatter(
                        df_exp_grp,
                        x="CIF_Medio",
                        y="Volume_Ton",
                        size="Operacoes",
                        color="Exportador_Grupo",
                        text="Exportador_Grupo",
                        title="Exportadores: Preço × Volume (tamanho = nº operações)",
                        color_discrete_sequence=ARKEMA_PALETTE,
                    )
                    fig_bubble.update_traces(textposition="top center", textfont=dict(size=9))
                    fig_bubble = apply_arkema_layout(fig_bubble, height=520)
                    fig_bubble.update_xaxes(title_text="CIF Médio (US$/kg)")
                    fig_bubble.update_yaxes(title_text="Volume Total (ton)")
                    st.plotly_chart(fig_bubble, use_container_width=True)

                    st.dataframe(
                        df_exp_grp.style.format({
                            "CIF_Medio": "US$ {:,.4f}",
                            "Volume_Ton": "{:,.1f} ton",
                        }),
                        use_container_width=True,
                    )

            with tab3:
                pais_col = "País" if "País" in df_filtrado.columns else ("País_Aquisição" if "País_Aquisição" in df_filtrado.columns else None)
                if pais_col:
                    df_pais = (
                        df_filtrado.groupby(pais_col)
                        .agg(Volume_Ton=('Peso_Ton', 'sum'), Valor_CIF=('Valor_CIF', 'sum'))
                        .reset_index()
                        .sort_values("Volume_Ton", ascending=False)
                    )
                    fig_pais = px.bar(
                        df_pais.head(20), x=pais_col, y="Volume_Ton",
                        title="Volume Importado por País de Origem",
                        color="Volume_Ton",
                        color_continuous_scale=["#45416A", "#70C0A7"],
                        text="Volume_Ton",
                    )
                    fig_pais.update_traces(
                        texttemplate="%{text:.1f} ton",
                        textposition="outside",
                        textfont=dict(size=9),
                    )
                    fig_pais = apply_arkema_layout(fig_pais, height=460)
                    fig_pais.update_xaxes(tickangle=-40, title_text="")
                    fig_pais.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig_pais, use_container_width=True)

    # =====================================================
    # 🔮 PREVISÃO (PROPHET)
    # =====================================================
    elif menu == "Previsão":
        st.markdown('<div class="arkema-red-bar"></div>', unsafe_allow_html=True)
        st.subheader("🔮 Previsão de Séries Temporais (Valores em US$)")

        if not df_filtrado.empty:
            available_metrics = [m for m in ["CIF_Unitário", "Peso_Ton", "Valor_CIF"] if m in df_filtrado.columns]
            metric_labels = {
                "CIF_Unitário": "CIF Unitário (US$/kg)",
                "Peso_Ton": "Volume (ton)",
                "Valor_CIF": "Valor CIF Total (US$)",
            }
            metrica = st.selectbox(
                "Selecione a métrica para prever:",
                available_metrics,
                format_func=lambda x: metric_labels.get(x, x),
                index=0,
            )
            periods = st.slider("Meses para prever:", 1, 24, 6)

            if metrica == "CIF_Unitário":
                df_p = df_filtrado.groupby("ANO/MÊS").apply(
                    lambda x: x['Valor_CIF'].sum() / x['Peso'].sum() if x['Peso'].sum() > 0 else 0
                ).reset_index().rename(columns={"ANO/MÊS": "ds", 0: "y"})
            else:
                df_p = df_filtrado.groupby("ANO/MÊS")[metrica].sum().reset_index().rename(
                    columns={"ANO/MÊS": "ds", metrica: "y"}
                )

            df_p = df_p[df_p['y'] > 0].sort_values("ds")

            if len(df_p) >= 2:
                with st.spinner("Calculando previsão..."):
                    m = Prophet(yearly_seasonality=True, interval_width=0.95)
                    m.fit(df_p)
                    future = m.make_future_dataframe(periods=periods, freq='MS')
                    forecast = m.predict(future)

                    fig_forecast = plot_plotly(m, forecast)
                    unit_label = metric_labels.get(metrica, metrica)
                    fig_forecast.update_layout(
                        title=f"Previsão — {unit_label}",
                        yaxis_title=unit_label,
                        xaxis_title="Data",
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        height=500,
                    )
                    fig_forecast.update_traces(hovertemplate="Data: %{x}<br>Valor: %{y:.4f}")
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    st.subheader("📊 Componentes da Tendência")
                    import matplotlib
                    matplotlib.rcParams.update({'figure.facecolor': 'white'})
                    st.pyplot(m.plot_components(forecast))

                    if st.checkbox("Mostrar Diagnóstico de Erro (MAPE)"):
                        try:
                            df_cv = cross_validation(m, initial='365 days', period='90 days', horizon='180 days')
                            df_perf = performance_metrics(df_cv)
                            st.metric("Erro Médio (MAPE)", f"{df_perf['mape'].mean() * 100:.2f}%")
                        except:
                            st.info("Dados insuficientes para validação estatística completa.")
            else:
                st.error("Dados históricos insuficientes (mínimo de 2 meses).")
        else:
            st.info("Carregue dados e aplique filtros para ver a previsão.")

else:
    st.markdown("""
    <div style="text-align:center; padding:3rem; background:white; border-radius:8px; border:2px dashed #E5E7EB;">
        <p style="font-size:3rem; margin:0">📂</p>
        <h3 style="color:#374151;">Aguardando arquivo</h3>
        <p style="color:#6B7280;">Carregue um arquivo <strong>.xlsx</strong> ou <strong>.csv</strong> para iniciar a análise.</p>
    </div>
    """, unsafe_allow_html=True)