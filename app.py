import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import requests

# --- Configuration de la page ---
st.set_page_config(page_title="📈 MoulaChart", page_icon="📊", layout="wide")

st.title("📈 MoulaChart")
st.markdown("### Compare les performances de plusieurs entreprises via **Yahoo Finance**")

st.divider()

# --- Charger la liste des tickers dynamiquement ---
@st.cache_data
def get_sp500_tickers():
    """Récupère les tickers du S&P 500 depuis Wikipedia"""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text)[0]
    tickers = table["Symbol"].tolist()
    names = table["Security"].tolist()
    mapping = dict(zip(tickers, names))
    return tickers, mapping

tickers_list, tickers_names = get_sp500_tickers()

# --- Dictionnaires FR -> valeurs pour yfinance ---
PERIODS = {
    "1 mois": "1mo",
    "3 mois": "3mo",
    "6 mois": "6mois",
    "1 an": "1y",
    "2 ans": "2y",
    "5 ans": "5y",
    "10 ans": "10y",
    "Max": "max",
}

INTERVALS = {
    "1 jour": "1d",
    "1 semaine": "1wk",
    "1 mois": "1mo",
}

# --- Layout ---
col_form, col_chart = st.columns([1, 3], gap="large")

with col_form:
    st.markdown("## Paramètres d’analyse")
    st.write("Choisis les actifs, la période et les options d’affichage.")

    tickers = st.multiselect(
        "Tickers à comparer :",
        options=tickers_list,
        format_func=lambda x: f"{x} – {tickers_names.get(x, '')}",
        default=["AAPL", "MSFT"]
    )

    st.markdown("---")

    # Sélecteurs français
    period_label = st.selectbox("Période :", list(PERIODS.keys()), index=3)
    interval_label = st.selectbox("Intervalle :", list(INTERVALS.keys()), index=0)

    # Conversion vers les valeurs techniques
    period = PERIODS[period_label]
    interval = INTERVALS[interval_label]

    st.markdown("---")
    normalize = st.toggle("Normaliser les prix (base 100)", value=True)
    show_stats = st.checkbox("Afficher le tableau de performance", value=True)

with col_chart:
    if tickers:
        with st.spinner("📡 Chargement des données depuis Yahoo Finance..."):
            raw_data = yf.download(
                tickers,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False
            )

            if isinstance(raw_data.columns, pd.MultiIndex):
                data = raw_data["Close"]
            else:
                data = raw_data[["Close"]].rename(columns={"Close": tickers[0]})

        # --- Normalisation optionnelle ---
        if normalize:
            data_plot = data / data.iloc[0] * 100
            ylabel = "Performance (base 100)"
        else:
            data_plot = data
            ylabel = "Prix ($)"

        # --- Graphique principal ---
        st.markdown("## Évolution des tickers sélectionnés")
        fig = px.line(
            data_plot,
            title="Performance historique des actifs sélectionnés",
            labels={"value": ylabel, "Date": "Date"}
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_stats:
            st.markdown("---")
            st.markdown("## Statistiques de performance")
            st.write("Données calculées à partir de la période et de l’intervalle choisis.")

            perf = pd.DataFrame()
            for t in tickers:
                returns = data[t].pct_change().dropna()
                perf.loc[t, "Performance (%)"] = (data[t].iloc[-1] / data[t].iloc[0] - 1) * 100
                perf.loc[t, "Volatilité (%)"] = returns.std() * 100
                perf.loc[t, "Rendement moyen (%)"] = returns.mean() * 100
            st.dataframe(perf.style.format("{:.2f}"))
    else:
        st.warning("🟡 Sélectionne au moins un ticker à gauche pour commencer.")
