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

# Configurar logging do Prophet para silenciar warnings excessivos
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

st.title("📊 Análise de Importação - PMMA")

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
        "Valor CIF Unitário": "CIF_Original",  # Renomeado para evitar conflito com o novo cálculo
        "CIF Unitário": "CIF_Original",
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

    # --- Limpeza Numérica Robusta ---
    numeric_cols = ["Peso", "Valor_FOB", "Valor_CIF", "Qtd_Estatística", "FOB_Unitário"]
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Convertemos para string e removemos espaços para evitar o TypeError
                df[col] = df[col].astype(str).str.replace(" ", "", regex=False)
                
                def clean_currency_string(val):
                    val = str(val) # Garante que o valor seja string para a comparação abaixo
                    if "," in val and "." in val:
                        return val.replace(".", "").replace(",", ".")
                    elif "," in val:
                        return val.replace(",", ".")
                    return val
                
                df[col] = df[col].apply(clean_currency_string)
            
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # =====================================================
    # MUDANÇA: CÁLCULO DE CIF UNITÁRIO REALISTA
    # =====================================================
    # Calcula a coluna baseada na relação Valor CIF / Peso
    if "Valor_CIF" in df.columns and "Peso" in df.columns:
        df["CIF_Unitário"] = df.apply(
            lambda r: r["Valor_CIF"] / r["Peso"] if r["Peso"] > 0 else 0.0, axis=1
        )

    # --- Tratamento de NCM ---
    if "NCM" in df.columns:
        df["NCM"] = df["NCM"].apply(lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.','',1).isdigit() else str(x))

    # =====================================================
    # 🔍 SEÇÃO DE FILTROS LATERAL
    # =====================================================
    st.sidebar.header("🔍 Filtros de Busca")
    
    ncm_list = sorted(df["NCM"].dropna().unique().tolist()) if "NCM" in df.columns else []
    ncm_default = [n for n in ncm_list if "39061000" in n]
    sel_ncm = st.sidebar.multiselect("Filtrar por NCM:", options=ncm_list, default=ncm_default)
    
    df_filtrado = df.copy()
    if sel_ncm:
        df_filtrado = df_filtrado[df_filtrado["NCM"].isin(sel_ncm)]
    
    descricoes_list = sorted(df_filtrado["Descrição"].dropna().unique().tolist()) if "Descrição" in df_filtrado.columns else []
    sel_descricoes = st.sidebar.multiselect("Filtrar por Descrição:", options=descricoes_list)
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
    # 📈 ANÁLISE HISTÓRICA (CORES ARKEMA ATUALIZADAS)
    # =====================================================
    if menu == "Análise Histórica":
        st.subheader("📈 Painel de Análise Histórica")
        
        # Paleta extraída da sua imagem
        ARKEMA_PURPLE = "#45416A"  # O início do logo ("ARK")
        ARKEMA_TEAL = "#70C0A7"    # O final do logo ("EMA")
        ARKEMA_GREEN = "#2E8B72"   # O tom médio do gradiente
        
        if df_filtrado.empty:
            st.warning("Nenhum dado encontrado para os filtros selecionados.")
        else:
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                pais_col = "País" if "País" in df_filtrado.columns else "País_Aquisição"
                paises_options = sorted(df_filtrado[pais_col].dropna().unique().tolist())
                sel_paises = st.multiselect("Filtrar por País de Origem:", paises_options)
                if sel_paises:
                    df_filtrado = df_filtrado[df_filtrado[pais_col].isin(sel_paises)]
            
            with col_f2:
                group_opts = ["Nenhum"] + [c for c in ["Descrição", "País", "Importador", "Exportador", "Modal", "Incoterm", "NCM"] if c in df_filtrado.columns]
                group_by_col = st.selectbox("Agrupar evolução temporal por:", group_opts)

            # --- Preparação de Dados ---
            group_cols = ["ANO/MÊS"]
            if group_by_col != "Nenhum":
                group_cols.append(group_by_col)
            
            df_grouped = df_filtrado.groupby(group_cols).agg({'Peso': 'sum', 'Valor_CIF': 'sum'}).reset_index()
            df_grouped['CIF_Unitário'] = df_grouped.apply(lambda r: r['Valor_CIF'] / r['Peso'] if r['Peso'] > 0 else 0, axis=1)
            df_grouped = df_grouped.sort_values("ANO/MÊS")
            
            # Médias para tendência
            df_grouped['Peso_Media_Movel'] = df_grouped['Peso'].rolling(window=3).mean()
            df_grouped['CIF_Media_Movel'] = df_grouped['CIF_Unitário'].rolling(window=3).mean()

            # --- Linha 1: Gráficos de Evolução ---
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_peso = px.line(df_grouped, x="ANO/MÊS", y="Peso", 
                                  color=group_by_col if group_by_col != "Nenhum" else None,
                                  color_discrete_sequence=[ARKEMA_PURPLE, ARKEMA_TEAL, ARKEMA_GREEN],
                                  title="Evolução de Peso Líquido (kg)", markers=True)
                if group_by_col == "Nenhum":
                    fig_peso.add_scatter(x=df_grouped["ANO/MÊS"], y=df_grouped["Peso_Media_Movel"], 
                                       name="Tendência (Média 3m)", line=dict(dash='dash', color=ARKEMA_TEAL))
                st.plotly_chart(fig_peso, use_container_width=True)

            with col_g2:
                fig_cif_u = px.line(df_grouped, x="ANO/MÊS", y="CIF_Unitário", 
                                   color=group_by_col if group_by_col != "Nenhum" else None,
                                   color_discrete_sequence=[ARKEMA_PURPLE, ARKEMA_TEAL, ARKEMA_GREEN],
                                   title="Evolução CIF Unitário (US$/kg)", markers=True)
                if group_by_col == "Nenhum":
                    fig_cif_u.add_scatter(x=df_grouped["ANO/MÊS"], y=df_grouped["CIF_Media_Movel"], 
                                         name="Tendência (Média 3m)", line=dict(dash='dash', color=ARKEMA_TEAL))
                st.plotly_chart(fig_cif_u, use_container_width=True)

            # --- Linha 2: Volume por Empresa e Preço por Exportador ---
            st.markdown("---")
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                st.subheader("Volume por Importador")
                df_imp = df_filtrado.groupby("Importador")["Peso"].sum().reset_index()
                df_imp["Toneladas"] = df_imp["Peso"] / 1000
                df_imp = df_imp.sort_values("Toneladas", ascending=False).head(15)
                fig_imp = px.bar(df_imp, x="Toneladas", y="Importador", orientation='h',
                                 title="Top 15 Importadores (Toneladas)")
                fig_imp.update_traces(marker_color=ARKEMA_PURPLE) # Cor principal do logo
                st.plotly_chart(fig_imp, use_container_width=True)

            with col_v2:
                st.subheader("Preço Médio por Exportador")
                df_exp = df_filtrado.groupby("Exportador").agg({'Valor_CIF': 'sum', 'Peso': 'sum'}).reset_index()
                df_exp["CIF_Médio"] = df_exp["Valor_CIF"] / df_exp["Peso"]
                df_exp = df_exp[df_exp["Peso"] > 500].sort_values("CIF_Médio", ascending=True).head(15)
                fig_exp = px.bar(df_exp, x="CIF_Médio", y="Exportador", orientation='h',
                                 title="Preço Médio por Exportador (US$/kg)")
                fig_exp.update_traces(marker_color=ARKEMA_GREEN) # Cor de destaque (verde médio)
                st.plotly_chart(fig_exp, use_container_width=True)

            # --- Tabela Detalhada ---
            st.subheader("🔎 Detalhamento")
            cols_show = ["ANO/MÊS", "NCM", "Descrição", "País", "Peso", "CIF_Unitário", "Importador", "Exportador"]
            cols_available = [c for c in cols_show if c in df_filtrado.columns]
            st.dataframe(df_filtrado[cols_available].sort_values("ANO/MÊS", ascending=False).style.format({
                "CIF_Unitário": "US$ {:,.4f}", "Peso": "{:,.2f} kg"}, decimal=',', thousands='.'), use_container_width=True)

            # --- Tabela Detalhada ---
            st.subheader("🔎 Detalhamento dos Dados")
            cols_show = ["ANO/MÊS", "NCM", "Descrição", "País", "Peso", "CIF_Unitário", "Importador", "Exportador"]
            cols_available = [c for c in cols_show if c in df_filtrado.columns]
            df_display = df_filtrado[cols_available].sort_values("ANO/MÊS", ascending=False)
            st.dataframe(df_display.style.format({"CIF_Unitário": "US$ {:,.4f}", "Peso": "{:,.2f} kg"}, decimal=',', thousands='.'), use_container_width=True)

    # =====================================================
    # 🔮 PREVISÃO (PROPHET)
    # =====================================================
    elif menu == "Previsão":
        st.subheader("🔮 Previsão de Séries Temporais (Valores em US$)")
        
        if not df_filtrado.empty:
            available_metrics = [m for m in ["CIF_Unitário", "Peso", "Valor_CIF"] if m in df_filtrado.columns]
            metrica = st.selectbox("Selecione a métrica para prever:", available_metrics, index=0)
            periods = st.slider("Meses para prever:", 1, 24, 6)

            if metrica == "CIF_Unitário":
                df_p = df_filtrado.groupby("ANO/MÊS").apply(
                    lambda x: x['Valor_CIF'].sum() / x['Peso'].sum() if x['Peso'].sum() > 0 else 0
                ).reset_index().rename(columns={"ANO/MÊS": "ds", 0: "y"})
            else:
                df_p = df_filtrado.groupby("ANO/MÊS")[metrica].sum().reset_index().rename(columns={"ANO/MÊS": "ds", metrica: "y"})
            
            df_p = df_p[df_p['y'] > 0].sort_values("ds")

            if len(df_p) >= 2:
                with st.spinner("Calculando previsão..."):
                    m = Prophet(yearly_seasonality=True, interval_width=0.95)
                    m.fit(df_p)
                    future = m.make_future_dataframe(periods=periods, freq='MS')
                    forecast = m.predict(future)
                    
                    fig_forecast = plot_plotly(m, forecast)
                    unit_label = "US$/kg" if metrica == "CIF_Unitário" else ("kg" if metrica == "Peso" else "US$")
                    fig_forecast.update_layout(title=f"Previsão de {metrica} ({unit_label})", yaxis_title=unit_label, xaxis_title="Data")
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