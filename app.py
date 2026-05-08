import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
import warnings
import logging
import re

# =====================================================
# CONFIGURAÇÕES INICIAIS E ESTILO ARKEMA
# =====================================================
st.set_page_config(page_title="Inteligência de Mercado - Arkema", layout="wide")

# Silenciar warnings do Prophet
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

# Identidade Visual Arkema
ARKEMA_COLORS = {
    "primary": "#45416A",    # Roxo Arkema
    "secondary": "#2E8B72",  # Verde Médio
    "accent": "#70C0A7",     # Verde Água
    "background": "#F8F9FA",
    "white": "#FFFFFF"
}

st.markdown(f"""
    <style>
    .stApp {{ background-color: {ARKEMA_COLORS['background']}; }}
    .main-title {{ 
        color: {ARKEMA_COLORS['primary']}; 
        font-weight: 800; 
        border-bottom: 3px solid {ARKEMA_COLORS['accent']}; 
        padding-bottom: 10px;
        margin-bottom: 20px;
    }}
    .metric-card {{
        background-color: {ARKEMA_COLORS['white']};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid {ARKEMA_COLORS['primary']};
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }}
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# FUNÇÕES DE TRATAMENTO DE DADOS
# =====================================================

def format_ncm(ncm_val):
    """Exibe o NCM acompanhado do nome amigável do produto."""
    ncm_str = str(ncm_val).split('.')[0].strip()
    mapping = {
        "39061000": "39061000 – PMMA",
        "39069040": "39069040 – Poliacrilatos",
        "39069019": "39069019 – Outros Acrílicos"
    }
    return mapping.get(ncm_str, f"{ncm_str} – Outros")

def group_exporters(name):
    """Agrupa exportadores por grupos econômicos (ex: LX)."""
    if pd.isna(name): return "NÃO INFORMADO"
    name = str(name).upper()
    if any(x in name for x in ["LX ", "LX-MMS", "LX INTERNATIONAL"]): return "GRUPO LX"
    if "ARKEMA" in name: return "ARKEMA GROUP"
    if "MITSUBISHI" in name: return "MITSUBISHI CHEMICAL"
    if "SUMITOMO" in name: return "SUMITOMO CHEMICAL"
    return name

def clean_numeric_col(val):
    """Limpeza de strings para conversão numérica (Padrão BR/US)."""
    if isinstance(val, (int, float)): return float(val)
    if pd.isna(val): return 0.0
    s = str(val).replace("US$", "").replace(" ", "").strip()
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."): s = s.replace(".", "").replace(",", ".")
        else: s = s.replace(",", "")
    elif "," in s: s = s.replace(",", ".")
    try: return float(re.sub(r'[^\d.-]', '', s))
    except: return 0.0

# =====================================================
# CARREGAMENTO E PROCESSAMENTO
# =====================================================

st.markdown("<h1 class='main-title'>📊 Inteligência de Mercado PMMA</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Arraste ou carregue o arquivo de dados (Excel ou CSV)", type=["xlsx", "csv"])

if uploaded_file:
    # Leitura do arquivo
    try:
        if uploaded_file.name.endswith('xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='latin1')
    except Exception as e:
        st.error(f"Erro ao ler o ficheiro: {e}")
        st.stop()

    # Normalização de Nomes de Colunas
    df.columns = [c.strip().upper() for c in df.columns]
    col_map = {
        "PESO LÍQUIDO": "Peso", 
        "VALOR CIF TOTAL": "Valor_CIF", 
        "ANO/MÊS": "Data", 
        "NCM": "NCM", 
        "DESCRIÇÃO PRODUTO": "Descricao",
        "PROVÁVEL IMPORTADOR": "Importador", 
        "PROVÁVEL EXPORTADOR": "Exportador"
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Tratamento de Tipos
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors='coerce')
        df = df.dropna(subset=["Data"])
    
    for c in ["Peso", "Valor_CIF"]:
        if c in df.columns: df[c] = df[c].apply(clean_numeric_col)

    # Criação de Métricas Realistas
    df["CIF_Unitario_Calculado"] = df.apply(lambda r: r["Valor_CIF"] / r["Peso"] if r["Peso"] > 0 else 0, axis=1)
    df["Peso_Ton"] = df["Peso"] / 1000
    df["Exportador_Agrupado"] = df["Exportador"].apply(group_exporters)

    # =====================================================
    # SIDEBAR - FILTROS COM "SELECT ALL"
    # =====================================================
    st.sidebar.markdown(f"<h2 style='color:{ARKEMA_COLORS['primary']}'>Filtros</h2>", unsafe_allow_html=True)
    
    # Filtro NCM Nomeado
    ncm_opts = sorted(df["NCM"].unique())
    sel_ncm = st.sidebar.multiselect("NCM / Produto:", options=ncm_opts, format_func=format_ncm)
    
    df_f = df[df["NCM"].isin(sel_ncm)] if sel_ncm else df.copy()

    # Filtro Descrição com Select All
    desc_opts = sorted(df_f["Descricao"].dropna().unique())
    sel_all_desc = st.sidebar.checkbox("Selecionar todas as descrições", value=True)
    
    if sel_all_desc:
        sel_desc = desc_opts
    else:
        sel_desc = st.sidebar.multiselect("Palavras-chave na Descrição:", options=desc_opts)
    
    df_f = df_f[df_f["Descricao"].isin(sel_desc)]
    
    # Filtros de Players
    imp_opts = sorted(df_f["Importador"].dropna().unique())
    sel_imp = st.sidebar.multiselect("Importador (Company):", options=imp_opts)
    if sel_imp: df_f = df_f[df_f["Importador"].isin(sel_imp)]

    menu = st.sidebar.radio("Navegação:", ["Análise Comercial", "Previsão de Tendência"])

    # =====================================================
    # DASHBOARD COMERCIAL
    # =====================================================
    if menu == "Análise Comercial":
        # KPIs principais
        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        with c_kpi1:
            st.metric("Volume Total (Ton)", f"{df_f['Peso_Ton'].sum():,.0f}")
        with c_kpi2:
            avg_cif_market = df_f["Valor_CIF"].sum() / df_f["Peso"].sum() if df_f["Peso"].sum() > 0 else 0
            st.metric("CIF Médio de Mercado", f"US$ {avg_cif_market:.2f}/kg")
        with c_kpi3:
            st.metric("Nº de Importadores", f"{len(df_f['Importador'].unique())}")

        # --- LINHA 1: EVOLUÇÃO E MÉDIAS ---
        st.markdown("### 📈 Evolução de Volume e Preço")
        col1, col2 = st.columns(2)
        
        # Agrupamento Mensal
        ts_data = df_f.groupby(df_f["Data"].dt.to_period("M")).agg({"Peso_Ton": "sum", "Valor_CIF": "sum"}).reset_index()
        ts_data["Data"] = ts_data["Data"].dt.to_timestamp()
        ts_data["CIF_Medio_Mes"] = ts_data["Valor_CIF"] / (ts_data["Peso_Ton"] * 1000)
        
        with col1:
            fig1 = px.line(ts_data, x="Data", y="Peso_Ton", title="Volume Mensal (Toneladas)", 
                          markers=True, text=ts_data["Peso_Ton"].apply(lambda x: f"{x:.0f}"))
            # Adicionar linha de média
            avg_v = ts_data["Peso_Ton"].mean()
            fig1.add_hline(y=avg_v, line_dash="dot", line_color="orange", annotation_text=f"Média: {avg_v:.1f}")
            fig1.update_traces(line_color=ARKEMA_COLORS["primary"], textposition="top center")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.line(ts_data, x="Data", y="CIF_Medio_Mes", title="CIF Médio (US$/kg)", 
                          markers=True, text=ts_data["CIF_Medio_Mes"].apply(lambda x: f"{x:.2f}"))
            # Adicionar linha de média
            avg_c = ts_data["CIF_Medio_Mes"].mean()
            fig2.add_hline(y=avg_c, line_dash="dot", line_color="red", annotation_text=f"Média: {avg_c:.2f}")
            fig2.update_traces(line_color=ARKEMA_COLORS["secondary"], textposition="top center")
            st.plotly_chart(fig2, use_container_width=True)

        # --- LINHA 2: CONCORRÊNCIA E EXPORTADORES ---
        st.markdown("### 🏢 Análise de Concorrência e Exportadores")
        col3, col4 = st.columns(2)

        with col3:
            # Volume (Toneladas) × Company (Importador)
            imp_vol = df_f.groupby("Importador")["Peso_Ton"].sum().nlargest(10).reset_index()
            fig3 = px.bar(imp_vol, x="Peso_Ton", y="Importador", orientation='h', 
                         title="Volume (Ton) × Importador (Top 10)", 
                         color_discrete_sequence=[ARKEMA_COLORS["accent"]],
                         text_auto='.1f')
            fig3.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            # Probable Exporter × Price
            exp_price = df_f.groupby("Exportador_Agrupado")["CIF_Unitario_Calculado"].mean().reset_index()
            fig4 = px.scatter(exp_price, x="CIF_Unitario_Calculado", y="Exportador_Agrupado", 
                             size="CIF_Unitario_Calculado", title="CIF Médio (US$/kg) × Exportador",
                             color_discrete_sequence=[ARKEMA_COLORS["primary"]])
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### 📋 Tabela Detalhada")
        st.dataframe(df_f[["Data", "Importador", "Exportador_Agrupado", "Peso_Ton", "CIF_Unitario_Calculado"]].sort_values("Data", ascending=False), use_container_width=True)

    # =====================================================
    # PREVISÃO PROPHET
    # =====================================================
    elif menu == "Previsão de Tendência":
        st.subheader("🔮 Previsão Estatística (Próximos 6 meses)")
        metrica = st.selectbox("Escolha a métrica para projetar:", ["Peso_Ton", "CIF_Unitario_Calculado"])
        
        df_p = df_f.groupby("Data")[metrica].mean().reset_index().rename(columns={"Data": "ds", metrica: "y"})
        df_p = df_p[df_p['y'] > 0]

        if len(df_p) > 2:
            with st.spinner("Calculando tendência..."):
                m = Prophet(yearly_seasonality=True, interval_width=0.95)
                m.fit(df_p)
                future = m.make_future_dataframe(periods=6, freq='MS')
                forecast = m.predict(future)
                
                fig_p = plot_plotly(m, forecast)
                fig_p.update_layout(title=f"Projeção de {metrica}", plot_bgcolor="white")
                st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.warning("Dados históricos insuficientes para gerar uma previsão fiável.")

else:
    st.info("⬆️ Por favor, carregue o ficheiro de importações para começar.")