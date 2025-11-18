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
                # Tenta ler a aba 'Sheet1', mas fallback para a primeira aba se não existir
                df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
            except ValueError:
                st.warning("A aba 'Sheet1' não foi encontrada. Lendo a primeira aba da planilha.")
                df = pd.read_excel(uploaded_file, sheet_name=0)
        
        elif file_extension == "csv":
            # Lendo CSV com inferência de delimitador
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
            
        else:
            st.error("Tipo de arquivo não suportado.")
            st.stop()
            
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    # --- Normalização e Mapeamento de Colunas (Adaptado para a nova planilha) ---
    # 1. Normaliza os nomes de coluna removendo espaços laterais
    df.columns = [col.strip() for col in df.columns]

    column_mapping = {
        # Colunas Chave para Agregação
        "Peso líquido": "Peso",
        "VALOR FOB ESTIMADO TOTAL": "Valor_FOB",
        "VALOR CIF TOTAL": "Valor_CIF",
        "QTD Estatística": "Qtd_Estatística",
        "Qtd. de operações estimada": "Qtd_Estatística", # Causa do conflito
        
        # Colunas de Agrupamento
        "Descrição produto": "Produto",
        "PAIS DE ORIGEM": "País",
        "País de aquisição": "País_Aquisição",
        "URF de Entrada": "URF_Entrada",
        "PROVÁVEL IMPORTADOR": "Importador",
        "PROVÁVEL EXPORTADOR": "Exportador",
        "NCM's": "NCM",
        "NCM": "NCM", 
        "MODAL": "Modal",
        "Incoterm": "Incoterm", 
        
        # Colunas Unitárias e Secundárias
        "Valor CIF Unitário": "CIF_Unitário",
        "Valor FOB Estimado Unitário": "FOB_Unitário", 
    }
    
    renamed_cols = {}
    for k_orig, v_new in column_mapping.items():
        if k_orig in df.columns:
            renamed_cols[k_orig] = v_new

    # --- CRÍTICO: RESOLVER CONFLITO DE DUPLICAÇÃO DE COLUNAS ANTES DE RENOMEAR ---
    
    # Colunas originais que mapeiam para o mesmo destino "Qtd_Estatística"
    col_qte_est_1 = "QTD Estatística"
    col_qte_est_2 = "Qtd. de operações estimada"
    
    # Se ambas as colunas originais estiverem presentes, removemos a de nome mais simples (menos descritivo)
    if col_qte_est_1 in df.columns and col_qte_est_2 in df.columns:
        try:
            # Dropa a primeira coluna, mantendo apenas a segunda, para evitar a duplicação no rename
            df.drop(columns=[col_qte_est_1], inplace=True)
            # Remove a coluna da lista de renomeação para evitar erro
            if col_qte_est_1 in renamed_cols:
                del renamed_cols[col_qte_est_1]
            st.warning(f"Conflito de colunas resolvido: '{col_qte_est_1}' foi removida em favor de '{col_qte_est_2}'.")
        except Exception as e:
            st.error(f"Erro ao tentar resolver o conflito de colunas: {e}")

    # 2. Executa a renomeação com as colunas restantes
    df = df.rename(columns=renamed_cols)

    # 3. NOVO PASSO CRÍTICO: REMOVER COLUNAS DUPLICADAS RESIDUAIS GERADAS PELO MAPPEAMENTO
    # Este passo é a garantia final para o PyArrow/Streamlit.
    if not df.empty:
        df = df.loc[:, ~df.columns.duplicated(keep='first')]


    # --- Tratamento da data (Robusto) ---
    try:
        def safe_to_datetime(dt_val):
            if pd.isna(dt_val):
                return pd.NaT
            
            dt_str = str(dt_val)
            
            if '-' in dt_str or '/' in dt_str:
                return pd.to_datetime(dt_str, errors='coerce')
            
            try:
                # Se for número (float ou string) tenta converter para YYYYMM (ex: 202507)
                if '.' in dt_str and dt_str.replace('.', '', 1).isdigit():
                    dt_str = dt_str.split('.')[0]
                    
                if dt_str.isdigit() and len(dt_str) >= 6:
                     # Tenta o formato YYYYMM
                     return pd.to_datetime(dt_str, format="%Y%m", errors='coerce')
            except:
                pass
            return pd.NaT
        
        if "ANO/MÊS" in df.columns:
            df["ANO/MÊS"] = df["ANO/MÊS"].apply(safe_to_datetime)
            df.dropna(subset=["ANO/MÊS"], inplace=True)
            
            if not pd.api.types.is_datetime64_any_dtype(df["ANO/MÊS"]):
                df["ANO/MÊS"] = pd.to_datetime(df["ANO/MÊS"], errors='coerce')
                df.dropna(subset=["ANO/MÊS"], inplace=True)
        else:
             st.error("A coluna 'ANO/MÊS' essencial para a análise de série temporal não foi encontrada.")
             st.stop()
             
    except Exception as e:
        st.error(f"Erro no tratamento da coluna 'ANO/MÊS'. Verifique se as datas estão no formato YYYYMM ou YYYY-MM-DD. Erro: {e}")
        st.stop()

    # --- Verificação e Backfill de Colunas Essenciais ---
    if "Peso" not in df.columns:
        st.error("A coluna 'Peso' (mapeada de 'Peso líquido') essencial para a análise não foi encontrada.")
        st.stop()

    for col in ["Valor_FOB", "Valor_CIF", "Qtd_Estatística"]:
        if col not in df.columns:
            df[col] = 0.0
            st.warning(f"A coluna '{col}' estava ausente e foi preenchida com zero para evitar erros de processamento.")
        
    # --- Conversão das colunas de valor para numérico (Lida com o formato brasileiro) ---
    numeric_cols = ["Peso", "Valor_FOB", "Valor_CIF", "Qtd_Estatística", "CIF_Unitário", "FOB_Unitário"]
    for col in numeric_cols:
        if col in df.columns:
            try:
                # 1. Usa .copy() para garantir que a cópia para limpeza não gere warnings
                data_to_clean = pd.Series(df[col]).copy()
                
                # 2. Converte para string
                col_data = data_to_clean.astype(str)
                
                # 3. Reforço de limpeza: Remove pontos (milhar), substitui vírgulas por pontos (decimal)
                col_data = (
                    col_data
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .str.replace(" ", "", regex=False) # remove espaços em branco residuais
                )
                
                # 4. Converte para numérico. Qualquer falha (incluindo 'nan') vira NaN
                df[col] = pd.to_numeric(col_data, errors='coerce')
                
                # 5. Preenche NaN com 0 para o cálculo e define o tipo como float
                df[col] = df[col].fillna(0).astype(float)
                
            except Exception as e:
                # Log do erro de limpeza, e tenta fallback.
                st.warning(f"Erro DETALHADO ao processar a coluna {col} na limpeza: {e}. Executando fallback robusto.")
                try:
                    # Fallback com Series() forçado para evitar o TypeError
                    df[col] = pd.to_numeric(pd.Series(df[col]), errors='coerce').fillna(0).astype(float)
                except Exception as fallback_e:
                    st.error(f"Erro CRÍTICO no Fallback para a coluna {col}: {fallback_e}. Esta coluna será tratada como zero para permitir a continuidade.")
                    df[col] = 0.0 # Definir como 0.0 se tudo falhar

    # --- Menu lateral ---
    menu = st.sidebar.radio("Escolha a análise:", ["Análise Histórica", "Previsão"])

    # =====================================================
    # ANÁLISE HISTÓRICA
    # =====================================================
    if menu == "Análise Histórica":
        st.subheader("📈 Análise Histórica")

        # --- Filtros e Agrupamento ---
        st.subheader("⚙️ Filtros e Agrupamento")
        
        col1, col2 = st.columns(2)
        
        # Filtros
        with col1:
            produtos_options = df["Produto"].dropna().unique() if "Produto" in df.columns else []
            produtos = st.multiselect("Selecione os produtos:", produtos_options)
        
        with col2:
            paises_options = df["País"].dropna().unique() if "País" in df.columns else []
            paises = st.multiselect("Selecione os países:", paises_options)

        df_filtrado = df.copy()
        if produtos and "Produto" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["Produto"].isin(produtos)]
        if paises and "País" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["País"].isin(paises)]

        # Seletor de Agrupamento
        group_by_options = ["Nenhum"] + [
            col for col in ["Importador", "Exportador", "País_Aquisição", "NCM", "Modal", "URF_Entrada", "Incoterm"] 
            if col in df.columns
        ]
        
        group_by_col = st.selectbox(
            "Agrupar evolução por:", 
            group_by_options
        )

        if df_filtrado.empty:
            st.warning("Nenhum dado encontrado com os filtros selecionados.")
        else:
            # Etapa 1: Preparar o DataFrame para Agrupamento e Copiar
            df_groupby_ready = df_filtrado.copy()

            # Etapa 2: Configurar o Agrupamento
            group_cols = ["ANO/MÊS"]
            color_param = None
            if group_by_col != "Nenhum" and group_by_col in df_groupby_ready.columns:
                group_cols.append(group_by_col)
                df_groupby_ready[group_by_col] = df_groupby_ready[group_by_col].astype(str).fillna("Não Informado")
                color_param = group_by_col
            elif group_by_col != "Nenhum":
                 st.warning(f"A coluna de agrupamento '{group_by_col}' não foi encontrada.")
                 group_by_col = "Nenhum"
                 color_param = None
            
            # Etapa 3: Configurar o Dicionário de Agregação
            agg_dict = {
                "Peso": 'sum',
                "Valor_FOB": 'sum',
                "Valor_CIF": 'sum'
            }
            if "Qtd_Estatística" in df_groupby_ready.columns:
                agg_dict["Qtd_Estatística"] = 'sum'

            # --- ESTRATÉGIA DE MERGE COM LIMPEZA DE COLUNAS ---
            
            # 1. Colunas de Agrupamento + Colunas de Agregação
            cols_to_select = group_cols + list(agg_dict.keys())
            
            # 2. Seleciona apenas as colunas disponíveis no DataFrame e faz a limpeza máxima
            available_cols_for_agg = [col for col in cols_to_select if col in df_groupby_ready.columns]
            df_final_for_agg = df_groupby_ready[available_cols_for_agg].copy()
            df_final_for_agg = df_final_for_agg.reset_index(drop=True) 

            # 3. Cria o objeto GroupBy
            grouped_obj = df_final_for_agg.groupby(group_cols, as_index=False)
            
            # 4. Agrega a primeira coluna para iniciar o DataFrame agrupado
            first_agg_col = list(agg_dict.keys())[0]
            
            # DataFrame inicial: Contém as colunas de agrupamento + a primeira agregação
            agrupado_agg = grouped_obj[first_agg_col].agg('sum').reset_index()

            # CRÍTICO: Limpar o DataFrame inicial de quaisquer colunas espúrias (como 'index' ou 'level_0')
            agrupado_agg = agrupado_agg[group_cols + [first_agg_col]].copy()

            # 5. Itera sobre o restante das colunas e junta (merge) ao DataFrame principal
            for col_to_agg in list(agg_dict.keys())[1:]: # Ignora a primeira
                func = agg_dict[col_to_agg]
                
                # 5a. Agrega a coluna restante e garante que o index seja resetado.
                temp_result = grouped_obj[col_to_agg].agg(func).reset_index()
                
                # CRÍTICO: Manter APENAS as colunas de join (group_cols) e a coluna agregada (col_to_agg)
                temp_result = temp_result[group_cols + [col_to_agg]]
                
                # 5b. Junta (merge) o resultado ao DataFrame principal (agrupado_agg)
                agrupado_agg = agrupado_agg.merge(
                    temp_result, 
                    on=group_cols, 
                    how='left'
                )
            
            # PASSO DE SEGURANÇA: LIMPEZA DE COLUNAS DUPLICADAS APÓS AGREGAÇÃO
            # Para evitar o erro narwhals/plotly.
            agrupado_agg = agrupado_agg.loc[:, ~agrupado_agg.columns.duplicated(keep='first')]


            # Garantir que a coluna 'ANO/MÊS' seja datetime para o Plotly
            agrupado_agg["ANO/MÊS"] = pd.to_datetime(agrupado_agg["ANO/MÊS"])
            
            # --- Plotagem de Quantidades ---
            st.subheader("📦 Evolução Quantidades")
            
            qty_cols = ["Peso"]
            
            if "Qtd_Estatística" in agrupado_agg.columns:
                try:
                    total_sum = agrupado_agg["Qtd_Estatística"].sum()
                    
                    if isinstance(total_sum, pd.Series) or isinstance(total_sum, np.ndarray):
                        if total_sum.item() > 0:
                            qty_cols.append("Qtd_Estatística")
                    elif total_sum > 0:
                        qty_cols.append("Qtd_Estatística")
                        
                except Exception:
                    st.warning("Não foi possível validar a soma total de 'Qtd_Estatística'. A coluna foi ignorada na plotagem de Quantidades.")
            
            
            if color_param:
                df_melted_qtd = agrupado_agg.melt(
                    id_vars=group_cols, 
                    value_vars=qty_cols, 
                    var_name="Métrica", 
                    value_name="Quantidade"
                )
                fig_qtd = px.line(
                    df_melted_qtd,
                    x="ANO/MÊS",
                    y="Quantidade",
                    color=color_param,
                    line_dash="Métrica",
                    labels={"Métrica": "Tipo de Quantidade", "ANO/MÊS": "Data"},
                    markers=True,
                    title=f"Evolução Mensal: Quantidades por {group_by_col}"
                )
            else:
                fig_qtd = px.line(
                    agrupado_agg,
                    x="ANO/MÊS",
                    y=qty_cols,
                    labels={"value": "Quantidade", "variable": "Métrica", "ANO/MÊS": "Data"},
                    markers=True,
                    title="Evolução Mensal: Quantidades (Peso Líquido vs. Estatística)"
                )

            fig_qtd.update_layout(hovermode="x unified")
            st.plotly_chart(fig_qtd, use_container_width=True)

            # --- Plotagem de Valores ---
            st.subheader("💰 Evolução Valores")
            
            value_cols = []
            if "Valor_FOB" in agrupado_agg.columns and agrupado_agg["Valor_FOB"].sum() > 0:
                 value_cols.append("Valor_FOB")
            if "Valor_CIF" in agrupado_agg.columns and agrupado_agg["Valor_CIF"].sum() > 0:
                 value_cols.append("Valor_CIF")
            
            if not value_cols:
                st.info("As colunas 'Valor_FOB' e 'Valor_CIF' estão ausentes ou contêm apenas zeros. Não é possível plotar a Evolução de Valores.")
            else:
                if color_param:
                    df_melted_valor = agrupado_agg.melt(
                        id_vars=group_cols, 
                        value_vars=value_cols, 
                        var_name="Métrica", 
                        value_name="Valor (US$)"
                    )
                    fig_valor = px.line(
                        df_melted_valor,
                        x="ANO/MÊS",
                        y="Valor (US$)",
                        color=color_param,
                        line_dash="Métrica",
                        labels={"Métrica": "Tipo de Valor", "ANO/MÊS": "Data"},
                        markers=True,
                        title=f"Evolução Mensal: Valor FOB e CIF por {group_by_col}"
                    )
                else:
                    fig_valor = px.line(
                        agrupado_agg,
                        x="ANO/MÊS",
                        y=value_cols,
                        labels={"value": "Valor (US$)", "variable": "Métrica", "ANO/MÊS": "Data"},
                        markers=True,
                        title="Evolução Mensal: Valor FOB vs. Valor CIF"
                    )
    
                fig_valor.update_layout(hovermode="x unified")
                st.plotly_chart(fig_valor, use_container_width=True)

            # --- Estatísticas ---
            st.subheader("📌 Estatísticas Resumidas")
            
            if not agrupado_agg.empty:
                desc_df = agrupado_agg.describe().T
                formatted_desc = desc_df.copy()
                for col in formatted_desc.columns:
                     if pd.api.types.is_numeric_dtype(formatted_desc[col]):
                        formatted_desc[col] = formatted_desc[col].apply(lambda x: f'{x:,.2f}')
                        
                st.dataframe(formatted_desc, use_container_width=True)
            else:
                 st.info("Não há dados para gerar as estatísticas resumidas.")


            # --- Visualização de Dados Detalhados ---
            st.subheader("🔎 Visualização dos Dados Detalhados")
            
            display_cols = [
                "ANO/MÊS", "Produto", "País", "Peso", "Valor_FOB", "Valor_CIF", "Qtd_Estatística",
                "FOB_Unitário", "CIF_Unitário", "País_Aquisição", "URF_Entrada", "Importador", "NCM", "Modal", "Exportador", "Incoterm"
            ]
            
            # Filtra apenas as colunas disponíveis no DataFrame filtrado e deduplicado
            # df_filtrado já é uma cópia de df, que foi deduplicado no início
            available_cols = [col for col in display_cols if col in df_filtrado.columns]

            st.dataframe(
                df_filtrado[available_cols].sort_values(by="ANO/MÊS", ascending=False), 
                use_container_width=True
            )

    # =====================================================
    # PREVISÃO COM PROPHET
    # =====================================================
    elif menu == "Previsão":
        st.subheader("🔮 Previsão de Séries Temporais")

        # Define as métricas disponíveis (apenas as que têm valores > 0)
        available_metrics = [col for col in ["Peso", "Qtd_Estatística", "Valor_FOB", "Valor_CIF"] if col in df.columns and df[col].sum() > 0]
        
        if not available_metrics:
            st.error("Nenhuma coluna numérica com valores maiores que zero foi encontrada para realizar a previsão. Por favor, verifique as colunas de Peso, Valor FOB ou Valor CIF.")
            st.stop()
            
        # Escolher métrica
        metrica = st.selectbox(
            "Selecione a métrica para previsão:",
            available_metrics
        )
        
        # Parâmetros do Prophet
        st.markdown("---")
        st.subheader("🛠️ Ajustes do Modelo Prophet (Hiperparâmetros)")

        col_params_1, col_params_2 = st.columns(2)
        
        with col_params_1:
            seasonality_mode = st.selectbox(
                "Modo de Sazonalidade:",
                ["multiplicative", "additive"],
                index=0,
                help="Multiplicativo: Sazonalidade cresce com a tendência. Aditivo: Sazonalidade constante."
            )
        with col_params_2:
            changepoint_prior_scale = st.slider(
                "Prior Scale (Flexibilidade da Tendência):",
                min_value=0.001,
                max_value=0.5,
                value=0.05,
                step=0.005,
                help="Maior valor = Modelo mais flexível/propenso a overfitting. Menor valor = Modelo mais suave."
            )

        # Agrupamento mensal
        agg_dict = {m: 'sum' for m in ["Peso", "Qtd_Estatística", "Valor_FOB", "Valor_CIF"] if m in df.columns}
        # Agrupamento simples para a Previsão (não usa os filtros de agrupamento)
        agrupado = df.groupby("ANO/MÊS").agg(agg_dict).reset_index()

        if agrupado.empty or metrica not in agrupado.columns:
            st.warning("Não há dados de série temporal suficientes ou a métrica selecionada não está disponível após o agrupamento.")
            st.stop()
        else:
            # Preparar dados para o Prophet
            df_prophet = agrupado[["ANO/MÊS", metrica]].rename(columns={"ANO/MÊS": "ds", metrica: "y"})
            df_prophet.dropna(inplace=True)

            if len(df_prophet) < 2:
                st.error("Dados insuficientes para a previsão. Pelo menos 2 pontos temporais são necessários.")
                st.stop()

            # --- Treinamento do Modelo ---
            with st.spinner(f"Treinando o modelo Prophet com a métrica {metrica}..."):
                model = Prophet(
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=changepoint_prior_scale,
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=True 
                )
                try:
                    model.fit(df_prophet)
                except Exception as e:
                    st.error(f"Erro ao treinar o modelo. Verifique a qualidade dos dados. Erro: {e}")
                    st.stop()


            # --- Previsão ---
            periods = st.slider("Selecione o número de meses para a previsão:", min_value=1, max_value=24, value=6)
            future = model.make_future_dataframe(periods=periods, freq="M")
            forecast = model.predict(future)

            # Plot interativo
            st.subheader(f"Gráfico de Previsão para {metrica}")
            fig_forecast = plot_plotly(model, forecast)
            fig_forecast.update_layout(
                title=f"Previsão de Importação - {metrica}",
                xaxis_title="Data",
                yaxis_title=f"{metrica} (Valor Previsto)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.subheader("📌 Componentes da Previsão")
            st.markdown("Os gráficos abaixo mostram a Tendência, Sazonalidade Anual e Pontos de Mudança detectados pelo modelo.")
            
            # Usando Matplotlib para plotar componentes
            fig_components = model.plot_components(forecast)
            st.pyplot(fig_components)
            plt.close(fig_components)
            
            st.subheader("📌 Tabela de Previsão")
            # Mostrar os últimos registros (histórico + previsão)
            st.dataframe(
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                .rename(columns={"ds": "Data", "yhat": "Previsão", "yhat_lower": "Limite Inferior", "yhat_upper": "Limite Superior"})
                .tail(periods + 1)
                .style.format(
                    {
                        "Previsão": "{:,.2f}",
                        "Limite Inferior": "{:,.2f}",
                        "Limite Superior": "{:,.2f}",
                        "Data": lambda x: x.strftime('%Y-%m-%d')
                    }
                ),
                use_container_width=True
            )

            # --- Cross-Validation ---
            st.markdown("---")
            st.subheader("🧪 Diagnóstico: Avaliação de Performance")
            
            min_data_points = 12 * 3
            if len(df_prophet) < min_data_points:
                st.info(f"O modelo Prophet sugere pelo menos {min_data_points} pontos de dados (meses) para uma validação cruzada robusta. Você tem apenas {len(df_prophet)}.")
            
            perform_cv = st.checkbox("Executar Validação Cruzada (Cross-Validation)?", value=False)
            
            if perform_cv:
                
                initial_months = max(int(len(df_prophet) * 0.5), 24)
                
                if initial_months >= len(df_prophet) - periods:
                    initial_months = max(len(df_prophet) - periods, 12)
                    if initial_months < 12:
                         st.warning("Dados insuficientes para uma CV significativa. Tentando com o mínimo possível.")
                         initial_months = max(len(df_prophet) - 3, 3)
                         
                h = f'{periods} months'
                initial = f'{initial_months} months'
                period_months = min(12, int((len(df_prophet) - initial_months) / 2))
                period = f'{period_months} months' if period_months > 0 else '6 months'


                st.info(f"Parâmetros da Validação Cruzada: Initial={initial}, Period={period}, Horizon={h}")
                
                try:
                    with st.spinner("Executando a Validação Cruzada (pode demorar)..."):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            df_cv = cross_validation(
                                model, 
                                initial=initial, 
                                period=period, 
                                horizon=h,
                                parallel="processes"
                            )
                        
                        df_p = performance_metrics(df_cv)

                    st.success("Validação Cruzada Concluída!")
                    
                    st.markdown("Métricas de performance médias ao longo do horizonte de previsão:")
                    st.dataframe(df_p[['horizon', 'rmse', 'mae', 'mape', 'mdape']].head(), use_container_width=True)

                    st.markdown("O RMSE (Root Mean Squared Error) e MAE (Mean Absolute Error) devem ser os menores possíveis. O MAPE (Mean Absolute Percentage Error) indica o erro percentual (idealmente < 10%).")
                    
                    fig_perf = px.line(
                        df_p,
                        x="horizon",
                        y=["rmse", "mape"],
                        title="Performance do Modelo (RMSE e MAPE) por Horizonte de Previsão",
                        labels={"value": "Métrica", "horizon": "Horizonte de Previsão"}
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)

                except ValueError as e:
                    st.error(f"Erro ao executar a Validação Cruzada. Verifique se a série temporal é longa o suficiente para os parâmetros de Initial={initial}, Period={period} e Horizon={h}. Erro: {e}")
                except Exception as e:
                    st.error(f"Ocorreu um erro inesperado durante a Validação Cruzada: {e}")
            else:
                st.info("A Validação Cruzada testa a precisão do modelo usando dados históricos, fornecendo métricas de erro como RMSE e MAPE.")

else:
    # Mensagem de instrução
    st.info("⬆️ Faça upload da planilha para iniciar a análise.")
    st.markdown("""
        **Estrutura esperada da planilha (colunas essenciais para o App funcionar):**
        - `ANO/MÊS`: Datas no formato `YYYYMM` (Ex: 202301, 202302) ou `YYYY-MM-DD`.
        - `Descrição produto`: Nome do produto.
        - `PAIS DE ORIGEM` ou `PAÍS DE ORIGEM`: Nome do país.
        - **`Peso líquido` (ESSENCIAL)**: Peso da importação.
        - `VALOR FOB ESTIMADO TOTAL`: Valor FOB total.
        - `VALOR CIF TOTAL`: Valor CIF total.
        - `QTD Estatística` ou `Qtd. de operações estimada`: Quantidade estatística.
        
        **Colunas adicionais suportadas (para agrupamento e detalhamento):**
        - `Incoterm` (novo para agrupamento), `Valor CIF Unitário`, `Valor FOB Estimado Unitário`, `País de aquisição`, `URF de Entrada`, `PROVÁVEL IMPORTADOR`, `NCM`, `MODAL`, `PROVÁVEL EXPORTADOR`
    """)