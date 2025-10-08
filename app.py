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
def get_tickers(source="S&P 500"):
    """Récupère les tickers selon la source choisie"""

    if source == "S&P 500":
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        table = pd.read_html(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text)[0]
        tickers = table["Symbol"].tolist()
        names = table["Security"].tolist()

    elif source == "NASDAQ 100":
        url = "https://en.wikipedia.org/wiki/NASDAQ-100"
        table = pd.read_html(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text)[4]
        tickers = table["Ticker"].tolist()
        names = table["Company"].tolist()

    elif source == "CAC 40":
        url = "https://en.wikipedia.org/wiki/CAC_40"
        table = pd.read_html(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text)[4]
        tickers = table["Ticker"].tolist()
        names = table["Company"].tolist()
        # Ajouter le suffixe .PA pour Yahoo Finance (Paris)
        tickers = [t + ".PA" if not t.endswith(".PA") else t for t in tickers]

    mapping = dict(zip(tickers, names))
    return tickers, mapping


# --- Dictionnaires FR -> valeurs pour yfinance ---
PERIODS = {
    "1 mois": "1mo",
    "3 mois": "3mo",
    "6 mois": "6mo",
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
    st.markdown("## Paramètres d'analyse")
    st.write("Choisis les actifs, la période et les options d'affichage.")

    # Sélecteur de source
    source = st.selectbox(
        "Source des tickers :",
        options=["S&P 500", "NASDAQ 100", "CAC 40"],
        index=0
    )

    # Charger les tickers selon la source
    tickers_list, tickers_names = get_tickers(source)

    tickers = st.multiselect(
        "Tickers à comparer :",
        options=tickers_list,
        format_func=lambda x: f"{x} – {tickers_names.get(x, '')}",
        default=[tickers_list[0], tickers_list[1]] if len(tickers_list) > 1 else [tickers_list[0]]
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

    st.markdown("---")
    compare_index = st.checkbox("Comparer avec l'indice de référence", value=False)

    # Définir l'indice selon la source
    if compare_index:
        if source == "S&P 500":
            index_ticker = "SPY"
            index_name = "S&P 500 (SPY)"
        elif source == "NASDAQ 100":
            index_ticker = "QQQ"
            index_name = "NASDAQ 100 (QQQ)"
        elif source == "CAC 40":
            index_ticker = "^FCHI"
            index_name = "CAC 40 (^FCHI)"

with col_chart:
    if tickers:
        # Ajouter l'indice de référence si demandé
        tickers_to_download = tickers.copy()
        if compare_index:
            tickers_to_download.append(index_ticker)

        with st.spinner("📡 Chargement des données depuis Yahoo Finance..."):
            raw_data = yf.download(
                tickers_to_download,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False
            )

            if isinstance(raw_data.columns, pd.MultiIndex):
                data = raw_data["Close"]
            else:
                data = raw_data[["Close"]].rename(columns={"Close": tickers_to_download[0]})

        # --- Normalisation optionnelle ---
        if normalize:
            data_plot = data / data.iloc[0] * 100
            ylabel = "Performance (base 100)"
        else:
            data_plot = data
            ylabel = "Prix ($)"

        # --- Graphique principal ---
        st.markdown("## 📈 Évolution des tickers sélectionnés")

        # Renommer l'indice pour l'affichage
        data_plot_display = data_plot.copy()
        if compare_index and index_ticker in data_plot_display.columns:
            data_plot_display = data_plot_display.rename(columns={index_ticker: index_name})

        fig = px.line(
            data_plot_display,
            title="Performance historique des actifs sélectionnés",
            labels={"value": ylabel, "Date": "Date"}
        )

        # Mettre l'indice en pointillés si présent
        if compare_index:
            for trace in fig.data:
                if trace.name == index_name:
                    trace.line.dash = 'dash'
                    trace.line.width = 3

        st.plotly_chart(fig, use_container_width=True)

        # --- Graphiques supplémentaires en colonnes ---
        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 📊 Rendements quotidiens")
            returns_data = data.pct_change().dropna() * 100
            if compare_index and index_ticker in returns_data.columns:
                returns_data = returns_data.rename(columns={index_ticker: index_name})
            fig_returns = px.line(
                returns_data,
                title="Rendements quotidiens (%)",
                labels={"value": "Rendement (%)", "Date": "Date"}
            )
            st.plotly_chart(fig_returns, use_container_width=True)

        with col_right:
            st.markdown("### 📉 Volatilité glissante (30 jours)")
            rolling_vol = data.pct_change().rolling(window=30).std() * 100
            if compare_index and index_ticker in rolling_vol.columns:
                rolling_vol = rolling_vol.rename(columns={index_ticker: index_name})
            fig_vol = px.line(
                rolling_vol,
                title="Volatilité sur 30 jours (%)",
                labels={"value": "Volatilité (%)", "Date": "Date"}
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        # --- Graphique des volumes (si disponibles) ---
        if "Volume" in raw_data.columns.get_level_values(0):
            st.markdown("---")
            st.markdown("### 📦 Volume de transactions")
            if isinstance(raw_data.columns, pd.MultiIndex):
                volume_data = raw_data["Volume"]
                if compare_index and index_ticker in volume_data.columns:
                    volume_data = volume_data.rename(columns={index_ticker: index_name})
                fig_volume = px.bar(
                    volume_data,
                    title="Volume de transactions",
                    labels={"value": "Volume", "Date": "Date"}
                )
                st.plotly_chart(fig_volume, use_container_width=True)

        # --- Graphique de corrélation ---
        if len(tickers) > 1:
            st.markdown("---")
            st.markdown("### 🔗 Matrice de corrélation")
            correlation = data.pct_change().corr()
            if compare_index and index_ticker in correlation.columns:
                correlation = correlation.rename(columns={index_ticker: index_name}, index={index_ticker: index_name})
            fig_corr = px.imshow(
                correlation,
                text_auto=".2f",
                title="Corrélation entre les rendements",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        if show_stats:
            st.markdown("---")
            st.markdown("## 📊 Statistiques de performance détaillées")
            st.write("Données calculées à partir de la période et de l'intervalle choisis.")

            perf = pd.DataFrame()
            tickers_for_stats = tickers_to_download if compare_index else tickers

            for t in tickers_for_stats:
                returns = data[t].pct_change().dropna()
                cumulative_returns = (1 + returns).cumprod()
                running_max = data[t].cummax()
                drawdown = (data[t] / running_max - 1) * 100

                # Informations de base
                perf.loc[t, "Prix initial ($)"] = data[t].iloc[0]
                perf.loc[t, "Prix final ($)"] = data[t].iloc[-1]
                perf.loc[t, "Plus haut ($)"] = data[t].max()
                perf.loc[t, "Plus bas ($)"] = data[t].min()

                # Performance
                perf.loc[t, "Performance totale (%)"] = (data[t].iloc[-1] / data[t].iloc[0] - 1) * 100
                perf.loc[t, "Performance annualisée (%)"] = ((data[t].iloc[-1] / data[t].iloc[0]) ** (
                            252 / len(data[t])) - 1) * 100

                # Risque
                perf.loc[t, "Volatilité quotidienne (%)"] = returns.std() * 100
                perf.loc[t, "Volatilité annualisée (%)"] = returns.std() * 100 * (252 ** 0.5)

                # Rendements
                perf.loc[t, "Rendement moyen (%)"] = returns.mean() * 100
                perf.loc[t, "Rendement médian (%)"] = returns.median() * 100

                # Ratios de performance
                risk_free_rate = 0.02 / 252  # Taux sans risque ~2% annuel
                excess_returns = returns - risk_free_rate
                perf.loc[t, "Ratio Sharpe"] = (returns.mean() / returns.std()) * (
                            252 ** 0.5) if returns.std() > 0 else 0
                perf.loc[t, "Ratio Sortino"] = (returns.mean() / returns[returns < 0].std()) * (252 ** 0.5) if len(
                    returns[returns < 0]) > 0 and returns[returns < 0].std() > 0 else 0

                # Drawdown
                perf.loc[t, "Drawdown max (%)"] = drawdown.min()
                perf.loc[t, "Drawdown actuel (%)"] = drawdown.iloc[-1]

                # Distribution des rendements
                perf.loc[t, "Skewness"] = returns.skew()
                perf.loc[t, "Kurtosis"] = returns.kurtosis()

                # Statistiques gagnant/perdant
                positive_returns = returns[returns > 0]
                negative_returns = returns[returns < 0]
                perf.loc[t, "Jours positifs (%)"] = (len(positive_returns) / len(returns)) * 100
                perf.loc[t, "Gain moyen (%)"] = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
                perf.loc[t, "Perte moyenne (%)"] = negative_returns.mean() * 100 if len(negative_returns) > 0 else 0
                perf.loc[t, "Meilleur jour (%)"] = returns.max() * 100
                perf.loc[t, "Pire jour (%)"] = returns.min() * 100
                perf.loc[t, "Ratio Gain/Perte"] = abs(positive_returns.mean() / negative_returns.mean()) if len(
                    negative_returns) > 0 and negative_returns.mean() != 0 else 0

            # Renommer l'indice dans le tableau
            if compare_index and index_ticker in perf.index:
                perf = perf.rename(index={index_ticker: index_name})

            # Afficher le tableau complet avec onglets
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📈 Vue d'ensemble", "💰 Prix & Performance", "⚠️ Risque & Volatilité", "📊 Distribution"])

            with tab1:
                overview_cols = ["Performance totale (%)", "Performance annualisée (%)", "Volatilité annualisée (%)",
                                 "Ratio Sharpe", "Drawdown max (%)", "Jours positifs (%)"]
                st.dataframe(
                    perf[overview_cols].style.format("{:.2f}")
                    .background_gradient(cmap="RdYlGn", subset=["Performance totale (%)", "Performance annualisée (%)"],
                                         axis=0)
                    .background_gradient(cmap="RdYlGn_r", subset=["Volatilité annualisée (%)", "Drawdown max (%)"],
                                         axis=0)
                    .background_gradient(cmap="RdYlGn", subset=["Ratio Sharpe", "Jours positifs (%)"], axis=0),
                    use_container_width=True
                )

            with tab2:
                price_cols = ["Prix initial ($)", "Prix final ($)", "Plus haut ($)", "Plus bas ($)",
                              "Performance totale (%)", "Performance annualisée (%)", "Rendement moyen (%)",
                              "Rendement médian (%)"]
                st.dataframe(
                    perf[price_cols].style.format("{:.2f}")
                    .background_gradient(cmap="Blues",
                                         subset=["Prix initial ($)", "Prix final ($)", "Plus haut ($)", "Plus bas ($)"],
                                         axis=0)
                    .background_gradient(cmap="RdYlGn", subset=["Performance totale (%)", "Performance annualisée (%)"],
                                         axis=0),
                    use_container_width=True
                )

            with tab3:
                risk_cols = ["Volatilité quotidienne (%)", "Volatilité annualisée (%)", "Ratio Sharpe", "Ratio Sortino",
                             "Drawdown max (%)", "Drawdown actuel (%)", "Meilleur jour (%)", "Pire jour (%)"]
                st.dataframe(
                    perf[risk_cols].style.format("{:.2f}")
                    .background_gradient(cmap="Reds",
                                         subset=["Volatilité quotidienne (%)", "Volatilité annualisée (%)"], axis=0)
                    .background_gradient(cmap="RdYlGn", subset=["Ratio Sharpe", "Ratio Sortino"], axis=0)
                    .background_gradient(cmap="Reds", subset=["Drawdown max (%)", "Drawdown actuel (%)"], axis=0),
                    use_container_width=True
                )

            with tab4:
                dist_cols = ["Jours positifs (%)", "Gain moyen (%)", "Perte moyenne (%)",
                             "Ratio Gain/Perte", "Skewness", "Kurtosis"]
                st.dataframe(
                    perf[dist_cols].style.format("{:.2f}")
                    .background_gradient(cmap="RdYlGn",
                                         subset=["Jours positifs (%)", "Gain moyen (%)", "Ratio Gain/Perte"], axis=0)
                    .background_gradient(cmap="RdYlGn_r", subset=["Perte moyenne (%)"], axis=0),
                    use_container_width=True
                )

            # --- Tableaux supplémentaires ---
            st.markdown("---")
            col_stat1, col_stat2 = st.columns(2)

            with col_stat1:
                st.markdown("### 🎯 Classement par performance")
                ranking = perf[["Performance totale (%)"]].sort_values("Performance totale (%)", ascending=False)
                st.dataframe(ranking.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=0),
                             use_container_width=True)

            with col_stat2:
                st.markdown("### ⚠️ Classement par risque (volatilité)")
                risk_ranking = perf[["Volatilité annualisée (%)"]].sort_values("Volatilité annualisée (%)",
                                                                               ascending=True)
                st.dataframe(risk_ranking.style.format("{:.2f}").background_gradient(cmap="RdYlGn_r", axis=0),
                             use_container_width=True)

            # --- Ratio rendement/risque ---
            st.markdown("---")
            st.markdown("### 📈 Ratio Rendement/Risque")
            perf["Ratio Rend/Risque"] = perf["Performance totale (%)"] / perf["Volatilité annualisée (%)"]
            ratio_ranking = perf[
                ["Performance totale (%)", "Volatilité annualisée (%)", "Ratio Rend/Risque"]].sort_values(
                "Ratio Rend/Risque", ascending=False)
            st.dataframe(
                ratio_ranking.style.format("{:.2f}").background_gradient(cmap="RdYlGn", subset=["Ratio Rend/Risque"],
                                                                         axis=0), use_container_width=True)

            # --- Graphique de dispersion rendement vs risque ---
            st.markdown("---")
            st.markdown("### 🎯 Rendement vs Risque")
            scatter_data = perf.reset_index()
            scatter_data.columns = ["Ticker"] + list(scatter_data.columns[1:])
            fig_scatter = px.scatter(
                scatter_data,
                x="Volatilité annualisée (%)",
                y="Performance totale (%)",
                text="Ticker",
                title="Rendement vs Risque",
                labels={"Volatilité annualisée (%)": "Risque (Volatilité %)",
                        "Performance totale (%)": "Rendement (%)"},
                size_max=20
            )
            fig_scatter.update_traces(textposition='top center', marker=dict(size=12))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Bouton de téléchargement CSV
        st.markdown("---")
        data_to_export = data.copy()
        if compare_index and index_ticker in data_to_export.columns:
            data_to_export = data_to_export.rename(columns={index_ticker: index_name})

        csv = data_to_export.to_csv().encode('utf-8')
        st.download_button(
            label="💾 Télécharger les données en CSV",
            data=csv,
            file_name=f"moulachart_data_{source.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )
    else:
        st.warning("🟡 Sélectionne au moins un ticker à gauche pour commencer.")