import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
import warnings
import logging
from matplotlib import pyplot as plt
import re

# Configurações iniciais
st.set_page_config(page_title="Análise de Importação PMMA", layout="wide")

logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

st.title("📊 Análise de Importação - PMMA")

# =====================================================
# MAPEAMENTO NCM → NOME DO PRODUTO (melhoria 5)
# =====================================================
NCM_NOMES = {
    "39061000": "39061000 – PMMA (Polimetilmetacrilato)",
    "39069090": "39069090 – Outros polímeros acrílicos",
    "39069010": "39069010 – Polímeros acrílicos em formas primárias",
    "39064000": "39064000 – Copolímeros de acrilonitrila",
    "32091000": "32091000 – Tintas e vernizes acrílicos",
    "29041011": "29041011 – Ácido metanossulfônico",
    "29309011": "29309011 – Outros compostos organossulfurados",
}

def get_ncm_label(ncm_val):
    return NCM_NOMES.get(str(ncm_val).strip(), str(ncm_val).strip())

# =====================================================
# AGRUPAMENTO DE EXPORTADORES (melhoria 4)
# =====================================================
EXPORT_GROUPS = {
    r"(?i)\bLX\b|ROHM|LUCITE|ALTUGLAS":  "LX / Röhm / Lucite",
    r"(?i)\bMITSUBISHI\b":               "Mitsubishi",
    r"(?i)\bSUMITOMO\b":                 "Sumitomo",
    r"(?i)\bCHIMEI\b":                   "Chi Mei",
    r"(?i)\bARKEMA\b":                   "Arkema",
    r"(?i)\bEVONIK\b":                   "Evonik",
    r"(?i)\bINEOS\b":                    "Ineos",
    r"(?i)\bDOW\b|\bDOWDUPONT\b":        "Dow",
    r"(?i)\bBASF\b":                     "BASF",
    r"(?i)\bSABIC\b":                    "SABIC",
    r"(?i)\bMERCK\b":                    "Merck",
    r"(?i)\bSIGMA\b|\bALDRICH\b":        "Sigma-Aldrich",
}

def agrupar_exportador(nome):
    if pd.isna(nome):
        return "Outros"
    for pattern, grupo in EXPORT_GROUPS.items():
        if re.search(pattern, str(nome)):
            return grupo
    return str(nome)[:40]

# --- Leitura do Arquivo (Suporte a Excel e CSV) ---
uploaded_file = st.file_uploader("Carregue o arquivo de dados (Excel ou CSV)", type=["xlsx", "csv"])
if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    df = pd.DataFrame()

    try:
        if file_extension == "xlsx":
            try:
                df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
            except ValueError:
                st.warning("A aba 'Sheet1' não foi encontrada. Lendo a primeira aba da planilha.")
                df = pd.read_excel(uploaded_file, sheet_name=0)
        elif file_extension == "csv":
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    # --- Normalização e Mapeamento de Colunas ---
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
        if "QTD Estatística" in renamed_cols: del renamed_cols["QTD Estatística"]

    df = df.rename(columns=renamed_cols)
    if not df.empty:
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # --- Tratamento da data ---
    def safe_to_datetime(dt_val):
        if pd.isna(dt_val): return pd.NaT
        dt_str = str(dt_val).strip()
        if '-' in dt_str or '/' in dt_str:
            return pd.to_datetime(dt_str, errors='coerce')
        try:
            if '.' in dt_str: dt_str = dt_str.split('.')[0]
            if dt_str.isdigit() and len(dt_str) >= 6:
                return pd.to_datetime(dt_str[:6], format="%Y%m", errors='coerce')
        except: pass
        return pd.NaT

    if "ANO/MÊS" in df.columns:
        df["ANO/MÊS"] = df["ANO/MÊS"].apply(safe_to_datetime)
        df.dropna(subset=["ANO/MÊS"], inplace=True)
    else:
        st.error("Coluna 'ANO/MÊS' não encontrada.")
        st.stop()

    # --- Limpeza Numérica Robusta (trata float, string e NaN) ---
    def clean_currency_string(val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        val = str(val).strip().replace(" ", "")
        if not val or val.lower() in ("nan", "none", "-"): return np.nan
        if "," in val and "." in val:
            return float(val.replace(".", "").replace(",", "."))
        elif "," in val:
            return float(val.replace(",", "."))
        try: return float(val)
        except ValueError: return np.nan

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
        # Melhoria 5: label com nome do produto
        df["NCM_Label"] = df["NCM"].apply(get_ncm_label)

    # Melhoria 1: CIF Unitário calculado como Valor_CIF / Peso (mais realista)
    if "Valor_CIF" in df.columns and "Peso" in df.columns:
        df["CIF_Unitário"] = df.apply(
            lambda r: r["Valor_CIF"] / r["Peso"] if r["Peso"] > 0 else np.nan, axis=1
        )

    # Peso em toneladas
    df["Peso_Ton"] = df["Peso"] / 1000 if "Peso" in df.columns else 0

    # Melhoria 4: agrupamento de exportadores
    if "Exportador" in df.columns:
        df["Exportador_Grupo"] = df["Exportador"].apply(agrupar_exportador)

    # =====================================================
    # 🔍 SEÇÃO DE FILTROS LATERAL
    # =====================================================
    st.sidebar.header("🔍 Filtros de Busca")

    # Melhoria 5: NCM com nome do produto
    if "NCM_Label" in df.columns:
        ncm_labels = sorted(df["NCM_Label"].dropna().unique().tolist())
        ncm_default = [n for n in ncm_labels if "39061000" in n]
        sel_ncm_labels = st.sidebar.multiselect("Filtrar por NCM:", options=ncm_labels, default=ncm_default)
        sel_ncm = [lbl.split(" – ")[0].strip() if " – " in lbl else lbl for lbl in sel_ncm_labels]
    else:
        ncm_list = sorted(df["NCM"].dropna().unique().tolist()) if "NCM" in df.columns else []
        ncm_default = [n for n in ncm_list if "39061000" in n]
        sel_ncm = st.sidebar.multiselect("Filtrar por NCM:", options=ncm_list, default=ncm_default)

    df_filtrado = df.copy()
    if sel_ncm:
        df_filtrado = df_filtrado[df_filtrado["NCM"].isin(sel_ncm)]

    # Melhoria 6: palavra-chave + select all na descrição
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
    menu = st.sidebar.radio("Navegação:", ["Análise Histórica", "Previsão"])

    # =====================================================
    # 📈 ANÁLISE HISTÓRICA
    # =====================================================
    if menu == "Análise Histórica":
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

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                # Melhoria 8: rótulos nos pontos
                fig_peso = px.line(
                    df_grouped, x="ANO/MÊS", y="Peso_Ton",
                    color=color_col,
                    title="Evolução de Volume (Toneladas)",
                    markers=True,
                    text="Peso_Ton",
                )
                fig_peso.update_traces(
                    texttemplate="%{text:.1f}",
                    textposition="top center",
                    textfont=dict(size=8),
                )
                # Melhoria 3: legenda abaixo para não comprimir o gráfico
                fig_peso.update_layout(
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                    yaxis_title="Volume (ton)",
                )
                # Melhoria 2: linha de média
                media_ton = df_grouped["Peso_Ton"].mean()
                fig_peso.add_hline(
                    y=media_ton, line_dash="dash", line_color="gray",
                    annotation_text=f"Média: {media_ton:,.1f} ton",
                    annotation_position="top right",
                )
                st.plotly_chart(fig_peso, use_container_width=True)

            with col_g2:
                fig_cif_u = px.line(
                    df_grouped, x="ANO/MÊS", y="CIF_Unitário",
                    color=color_col,
                    title="Evolução CIF Unitário (US$/kg)",
                    markers=True,
                    text="CIF_Unitário",
                )
                fig_cif_u.update_traces(
                    texttemplate="US$%{text:.2f}",
                    textposition="top center",
                    textfont=dict(size=8),
                    hovertemplate="Data: %{x}<br>CIF Unitário: US$ %{y:.4f}/kg",
                )
                fig_cif_u.update_layout(
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                    yaxis_title="US$/kg",
                )
                # Melhoria 2: linha de média
                media_cif = df_grouped["CIF_Unitário"].mean()
                fig_cif_u.add_hline(
                    y=media_cif, line_dash="dash", line_color="gray",
                    annotation_text=f"Média: US${media_cif:,.4f}/kg",
                    annotation_position="top right",
                )
                st.plotly_chart(fig_cif_u, use_container_width=True)

            # =====================================================
            # Melhoria 2: Novos gráficos — Volume × Empresa e Exportador × Preço
            # =====================================================
            st.subheader("🏢 Análise por Empresa")

            col_n1, col_n2 = st.columns(2)

            with col_n1:
                if "Importador" in df_filtrado.columns:
                    df_vol_emp = (
                        df_filtrado.groupby("Importador")["Peso_Ton"]
                        .sum().sort_values(ascending=False).head(15).reset_index()
                    )
                    fig_vol_emp = px.bar(
                        df_vol_emp, x="Importador", y="Peso_Ton",
                        title="Volume (Toneladas) × Importador",
                        text="Peso_Ton",
                    )
                    fig_vol_emp.update_traces(
                        texttemplate="%{text:.1f}t",
                        textposition="outside",
                        textfont=dict(size=9),
                    )
                    fig_vol_emp.update_layout(
                        xaxis_tickangle=-40,
                        yaxis_title="Volume (ton)",
                        xaxis_title="",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_vol_emp, use_container_width=True)

            with col_n2:
                if "Exportador_Grupo" in df_filtrado.columns:
                    df_exp_price = (
                        df_filtrado.groupby("Exportador_Grupo")
                        .apply(lambda x: x["Valor_CIF"].sum() / x["Peso"].sum() if x["Peso"].sum() > 0 else 0)
                        .reset_index(name="CIF_Medio")
                        .sort_values("CIF_Medio", ascending=False)
                    )
                    fig_exp = px.bar(
                        df_exp_price, x="Exportador_Grupo", y="CIF_Medio",
                        title="Exportador × Preço CIF Médio (US$/kg)",
                        text="CIF_Medio",
                    )
                    fig_exp.update_traces(
                        texttemplate="US$%{text:.2f}",
                        textposition="outside",
                        textfont=dict(size=9),
                    )
                    fig_exp.update_layout(
                        xaxis_tickangle=-40,
                        yaxis_title="CIF Médio (US$/kg)",
                        xaxis_title="",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_exp, use_container_width=True)

            # =====================================================
            # Detalhamento
            # =====================================================
            st.subheader("🔎 Detalhamento dos Dados")
            cols_show = ["ANO/MÊS", "NCM_Label", "Descrição", "País", "Peso_Ton", "CIF_Unitário", "Valor_CIF", "Importador", "Exportador_Grupo"]
            cols_available = [c for c in cols_show if c in df_filtrado.columns]

            df_display = df_filtrado[cols_available].sort_values("ANO/MÊS", ascending=False)
            fmt = {}
            if "CIF_Unitário" in df_display.columns: fmt["CIF_Unitário"] = "US$ {:,.4f}"
            if "Valor_CIF"    in df_display.columns: fmt["Valor_CIF"]    = "US$ {:,.2f}"
            if "Peso_Ton"     in df_display.columns: fmt["Peso_Ton"]     = "{:,.3f} ton"

            st.dataframe(
                df_display.style.format(fmt, decimal=',', thousands='.'),
                use_container_width=True
            )

    # =====================================================
    # 🔮 PREVISÃO (PROPHET)
    # =====================================================
    elif menu == "Previsão":
        st.subheader("🔮 Previsão de Séries Temporais (Valores em US$)")

        if not df_filtrado.empty:
            available_metrics = [m for m in ["CIF_Unitário", "Peso_Ton", "Valor_CIF"] if m in df_filtrado.columns]
            metric_labels = {
                "CIF_Unitário": "CIF Unitário (US$/kg)",
                "Peso_Ton":     "Volume (ton)",
                "Valor_CIF":    "Valor CIF Total (US$)",
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
                        title=f"Previsão de {unit_label}",
                        yaxis_title=unit_label,
                        xaxis_title="Data"
                    )
                    fig_forecast.update_traces(hovertemplate="Data: %{x}<br>Valor: %{y:.4f}")
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    st.subheader("📊 Componentes da Tendência")
                    st.pyplot(m.plot_components(forecast))

                    if st.checkbox("Mostrar Diagnóstico de Erro (MAPE)"):
                        try:
                            df_cv = cross_validation(m, initial='365 days', period='90 days', horizon='180 days')
                            df_perf = performance_metrics(df_cv)
                            st.write(f"Erro Médio (MAPE): {df_perf['mape'].mean() * 100:.2f}%")
                        except:
                            st.info("Dados insuficientes para validação estatística completa.")
            else:
                st.error("Dados históricos insuficientes para gerar previsão (mínimo de 2 meses históricos).")
        else:
            st.info("Carregue dados e aplique filtros para ver a previsão.")

else:
    st.info("⬆️ Aguardando upload do arquivo Excel ou CSV para iniciar a análise.")