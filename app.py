import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title="Análise de Importação PMMA", layout="wide")

st.title("📊 Análise de Importação - PMMA")

# Upload do arquivo
uploaded_file = st.file_uploader("Carregue a planilha Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Sheet1")

    # --- Tratamento da data ---
    df["ANO/MÊS"] = df["ANO/MÊS"].dropna().astype(int).astype(str)
    df["ANO/MÊS"] = pd.to_datetime(df["ANO/MÊS"], format="%Y%m")

    # --- Seleção de colunas relevantes ---
    df = df.rename(columns={
        "Descrição produto": "Produto",
        "PAIS DE ORIGEM": "País",
        "Peso líquido": "Peso",
        "VALOR FOB ESTIMADO TOTAL": "Valor_FOB",
        "VALOR CIF TOTAL": "Valor_CIF",
        "QTD Estatística": "Qtd_Estatística"
    })

    # Conversão da QTD Estatística para numérico
    df["Qtd_Estatística"] = (
        df["Qtd_Estatística"].astype(str).str.replace(",", ".").astype(float)
    )

    # --- Menu lateral ---
    menu = st.sidebar.radio("Escolha a análise:", ["Análise Histórica", "Previsão"])

    # =====================================================
    # ANÁLISE HISTÓRICA
    # =====================================================
    if menu == "Análise Histórica":
        st.subheader("📈 Análise Histórica")

        # --- Filtros ---
        produtos = st.multiselect("Selecione os produtos:", df["Produto"].dropna().unique())
        paises = st.multiselect("Selecione os países:", df["País"].dropna().unique())

        df_filtrado = df.copy()
        if produtos:
            df_filtrado = df_filtrado[df_filtrado["Produto"].isin(produtos)]
        if paises:
            df_filtrado = df_filtrado[df_filtrado["País"].isin(paises)]

        # --- Agrupamento por mês ---
        agrupado = df_filtrado.groupby("ANO/MÊS").agg({
            "Peso": "sum",
            "Qtd_Estatística": "sum",
            "Valor_FOB": "sum",
            "Valor_CIF": "sum"
        }).reset_index()

        st.subheader("📦 Evolução Quantidades")
        fig_qtd = px.line(
            agrupado,
            x="ANO/MÊS",
            y=["Peso", "Qtd_Estatística"],
            labels={"value": "Quantidade", "ANO/MÊS": "Data"},
            markers=True
        )
        st.plotly_chart(fig_qtd, use_container_width=True)

        st.subheader("💰 Evolução Valores")
        fig_valor = px.line(
            agrupado,
            x="ANO/MÊS",
            y=["Valor_FOB", "Valor_CIF"],
            labels={"value": "Valor (US$)", "ANO/MÊS": "Data"},
            markers=True
        )
        st.plotly_chart(fig_valor, use_container_width=True)

        # --- Estatísticas ---
        st.subheader("📌 Estatísticas Resumidas")
        st.write(agrupado.describe())

    # =====================================================
    # PREVISÃO COM PROPHET
    # =====================================================
    elif menu == "Previsão":
        st.subheader("🔮 Previsão de Séries Temporais")

        # Escolher métrica
        metrica = st.selectbox(
            "Selecione a métrica para previsão:",
            ["Peso", "Qtd_Estatística", "Valor_FOB", "Valor_CIF"]
        )

        # Agrupamento mensal
        agrupado = df.groupby("ANO/MÊS").agg({
            "Peso": "sum",
            "Qtd_Estatística": "sum",
            "Valor_FOB": "sum",
            "Valor_CIF": "sum"
        }).reset_index()

        # Preparar dados para o Prophet
        df_prophet = agrupado[["ANO/MÊS", metrica]].rename(columns={"ANO/MÊS": "ds", metrica: "y"})

        # Criar modelo
        model = Prophet()
        model.fit(df_prophet)

        # Previsão para 6 meses
        future = model.make_future_dataframe(periods=6, freq="M")
        forecast = model.predict(future)

        # Plot interativo
        fig_forecast = plot_plotly(model, forecast)
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.subheader("📌 Tabela de Previsão")
        st.write(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12))
else:
    st.info("⬆️ Faça upload da planilha para iniciar a análise")
