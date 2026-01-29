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

    # Ajuste: "Descrição produto" agora é mapeado para "Descrição"
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

    # --- Limpeza Numérica (Ajuste de vírgulas e pontos) ---
    numeric_cols = ["Peso", "Valor_FOB", "Valor_CIF", "Qtd_Estatística", "CIF_Unitário", "FOB_Unitário"]
    for col in numeric_cols:
        if col in df.columns:
            # Tratamento robusto: remove pontos de milhar e troca vírgula por ponto decimal
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False),
                errors='coerce'
            ).fillna(0).astype(float)

    # =====================================================
    # 🔍 SEÇÃO DE FILTROS LATERAL (ORDEM E LÓGICA ALTERADAS)
    # =====================================================
    st.sidebar.header("🔍 Filtros de Busca")
    
    # 1. NCM como primeiro filtro
    ncm_list = sorted(df["NCM"].dropna().unique().tolist()) if "NCM" in df.columns else []
    sel_ncm = st.sidebar.multiselect("Filtrar por NCM (Vazio = Todos):", options=ncm_list)
    
    # 2. Descrição (Função Guarda-Chuva baseada no NCM)
    if sel_ncm:
        desc_subset = df[df["NCM"].isin(sel_ncm)]["Descrição"]
    else:
        desc_subset = df["Descrição"]
    
    descricoes_list = sorted(desc_subset.dropna().unique().tolist()) if "Descrição" in df.columns else []
    sel_descricoes = st.sidebar.multiselect("Filtrar por Descrição (Vazio = Todos):", options=descricoes_list)
    
    # 3. Importadores
    importadores_list = sorted(df["Importador"].dropna().unique().tolist()) if "Importador" in df.columns else []
    sel_importadores = st.sidebar.multiselect("Pesquisar Importadores:", options=importadores_list)

    # 4. Exportadores
    exportadores_list = sorted(df["Exportador"].dropna().unique().tolist()) if "Exportador" in df.columns else []
    sel_exportadores = st.sidebar.multiselect("Pesquisar Exportadores:", options=exportadores_list)

    # Aplicação centralizada dos filtros
    df_filtrado = df.copy()
    if sel_ncm:
        df_filtrado = df_filtrado[df_filtrado["NCM"].isin(sel_ncm)]
    if sel_descricoes:
        df_filtrado = df_filtrado[df_filtrado["Descrição"].isin(sel_descricoes)]
    if sel_importadores:
        df_filtrado = df_filtrado[df_filtrado["Importador"].isin(sel_importadores)]
    if sel_exportadores:
        df_filtrado = df_filtrado[df_filtrado["Exportador"].isin(sel_exportadores)]

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
                paises_options = sorted(df_filtrado["País"].dropna().unique().tolist())
                sel_paises = st.multiselect("Filtrar por País de Origem:", paises_options)
                if sel_paises:
                    df_filtrado = df_filtrado[df_filtrado["País"].isin(sel_paises)]
            
            with col_f2:
                group_opts = ["Nenhum"] + [c for c in ["Descrição", "País", "Importador", "Exportador", "Modal", "Incoterm", "NCM"] if c in df_filtrado.columns]
                group_by_col = st.selectbox("Agrupar evolução temporal por:", group_opts)

            # Preparação de Dados Agrupados
            group_cols = ["ANO/MÊS"]
            if group_by_col != "Nenhum":
                group_cols.append(group_by_col)
            
            # Agregação para Totais e Médias Ponderadas
            df_grouped = df_filtrado.groupby(group_cols).agg({
                'Peso': 'sum',
                'Valor_FOB': 'sum',
                'Valor_CIF': 'sum',
                'Qtd_Estatística': 'sum'
            }).reset_index()
            
            # Cálculo do CIF Unitário (Ponderado pelo peso total do grupo/mês)
            df_grouped['CIF_Unitário'] = df_grouped.apply(
                lambda row: row['Valor_CIF'] / row['Peso'] if row['Peso'] > 0 else 0, axis=1
            )

            # --- Gráficos ---
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_peso = px.line(df_grouped, x="ANO/MÊS", y="Peso", color=group_by_col if group_by_col != "Nenhum" else None,
                                  title="Evolução de Peso Líquido (kg)", markers=True)
                st.plotly_chart(fig_peso, use_container_width=True)

            with col_g2:
                # Indicação clara de moeda (US$/kg) no título
                fig_cif_u = px.line(df_grouped, x="ANO/MÊS", y="CIF_Unitário", color=group_by_col if group_by_col != "Nenhum" else None,
                                   title="Evolução CIF Unitário (US$/kg)", markers=True)
                # Formatação das dicas (hover) para exibir vírgulas e moeda
                fig_cif_u.update_traces(hovertemplate="Data: %{x}<br>CIF Unitário: US$ %{y:.4f}/kg")
                st.plotly_chart(fig_cif_u, use_container_width=True)

            st.subheader("🔎 Detalhamento dos Dados")
            # Tabela formatada com moeda
            cols_show = ["ANO/MÊS", "NCM", "Descrição", "País", "Peso", "CIF_Unitário", "Importador", "Exportador"]
            cols_available = [c for c in cols_show if c in df_filtrado.columns]
            
            # Formatação visual da tabela para o usuário
            st.dataframe(
                df_filtrado[cols_available].sort_values("ANO/MÊS", ascending=False).style.format({
                    "CIF_Unitário": "US$ {:.4f}",
                    "Peso": "{:.2f} kg"
                }), 
                use_container_width=True
            )

    # =====================================================
    # 🔮 PREVISÃO (PROPHET)
    # =====================================================
    elif menu == "Previsão":
        st.subheader("🔮 Previsão de Séries Temporais (Valores em US$)")
        
        available_metrics = [m for m in ["CIF_Unitário", "Peso", "Valor_FOB", "Valor_CIF"] if m in df_filtrado.columns]
        metrica = st.selectbox("Selecione a métrica para prever:", available_metrics, index=0)

        with st.expander("🛠️ Ajustes do Modelo"):
            c1, c2 = st.columns(2)
            with c1:
                seasonality_mode = st.selectbox("Sazonalidade:", ["multiplicative", "additive"])
                periods = st.slider("Meses para prever:", 1, 24, 6)
            with c2:
                changepoint_scale = st.slider("Flexibilidade (Prior Scale):", 0.001, 0.5, 0.05, 0.005)

        # Agrupamento para Prophet
        if metrica == "CIF_Unitário":
            df_p = df_filtrado.groupby("ANO/MÊS").apply(
                lambda x: x['Valor_CIF'].sum() / x['Peso'].sum() if x['Peso'].sum() > 0 else 0
            ).reset_index().rename(columns={"ANO/MÊS": "ds", 0: "y"})
        else:
            df_p = df_filtrado.groupby("ANO/MÊS")[metrica].sum().reset_index().rename(columns={"ANO/MÊS": "ds", metrica: "y"})
        
        if len(df_p) >= 2:
            with st.spinner("Calculando previsão..."):
                m = Prophet(seasonality_mode=seasonality_mode, changepoint_prior_scale=changepoint_scale, yearly_seasonality=True)
                m.fit(df_p)
                future = m.make_future_dataframe(periods=periods, freq='M')
                forecast = m.predict(future)
                
                fig_forecast = plot_plotly(m, forecast)
                unit_label = "US$/kg" if metrica == "CIF_Unitário" else ("kg" if metrica == "Peso" else "US$")
                fig_forecast.update_layout(title=f"Previsão de {metrica} ({unit_label})")
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.subheader("📊 Componentes da Tendência")
                st.pyplot(m.plot_components(forecast))

                st.markdown("---")
                if st.checkbox("Executar Diagnóstico de Erro (Cross-Validation)?"):
                    try:
                        with st.spinner("Processando..."):
                            df_cv = cross_validation(m, initial='730 days', period='180 days', horizon=f'{periods*30} days')
                            df_perf = performance_metrics(df_cv)
                            st.write("Erro Médio (MAPE): {:.2f}%".format(df_perf['mape'].mean() * 100))
                            st.dataframe(df_perf[['horizon', 'rmse', 'mae', 'mape']])
                    except Exception:
                        st.info("Dados insuficientes para validação cruzada.")
        else:
            st.error("Dados insuficientes para gerar previsão (mínimo de 2 meses históricos).")

else:
    st.info("⬆️ Aguardando upload do arquivo Excel ou CSV para iniciar a análise.")