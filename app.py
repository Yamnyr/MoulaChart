import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# --- Configuration de la page ---
st.set_page_config(page_title="📈 MoulaChart", page_icon="📊", layout="wide")

# --- Styles CSS personnalisés ---
def load_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --- En-tête principal ---
# st.markdown("""
# <div class="main-header">
#     <h1>📈 MoulaChart</h1>
#     <p>Analyse avancée des performances boursières avec Yahoo Finance</p>
# </div>
# """, unsafe_allow_html=True)


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
        tickers = [t + ".PA" if not t.endswith(".PA") else t for t in tickers]

    mapping = dict(zip(tickers, names))
    return tickers, mapping


# --- Dictionnaires FR -> valeurs pour yfinance ---
PERIODS = {
    "1 mois": "1mo", "3 mois": "3mo", "6 mois": "6mo", "1 an": "1y",
    "2 ans": "2y", "5 ans": "5y", "10 ans": "10y", "Max": "max",
}

INTERVALS = {"1 jour": "1d", "1 semaine": "1wk", "1 mois": "1mo"}

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    source = st.selectbox(
        "📊 Source des données",
        options=["S&P 500", "NASDAQ 100", "CAC 40"],
        index=0
    )

    tickers_list, tickers_names = get_tickers(source)

    tickers = st.multiselect(
        "🎯 Actifs à analyser",
        options=tickers_list,
        format_func=lambda x: f"{x} – {tickers_names.get(x, '')}",
        default=[tickers_list[0], tickers_list[1]] if len(tickers_list) > 1 else [tickers_list[0]]
    )

    st.markdown("---")
    st.subheader("📅 Période d'analyse")

    col1, col2 = st.columns(2)
    with col1:
        period_label = st.selectbox("Période", list(PERIODS.keys()), index=3)
    with col2:
        interval_label = st.selectbox("Intervalle", list(INTERVALS.keys()), index=0)

    period = PERIODS[period_label]
    interval = INTERVALS[interval_label]

    st.markdown("---")
    st.subheader("🎨 Options d'affichage")

    normalize = st.toggle("📊 Normaliser (base 100)", value=True)
    show_stats = st.checkbox("📈 Tableau de performance", value=True)
    compare_index = st.checkbox("🔍 Comparer avec l'indice", value=False)

    if compare_index:
        if source == "S&P 500":
            index_ticker, index_name = "SPY", "S&P 500 (SPY)"
        elif source == "NASDAQ 100":
            index_ticker, index_name = "QQQ", "NASDAQ 100 (QQQ)"
        else:
            index_ticker, index_name = "^FCHI", "CAC 40 (^FCHI)"
    else:
        index_ticker = index_name = None

# --- Contenu principal ---
if tickers:
    tickers_to_download = tickers.copy()
    if compare_index and index_ticker:
        tickers_to_download.append(index_ticker)

    with st.spinner("🔄 Chargement des données..."):
        raw_data = yf.download(tickers_to_download, period=period, interval=interval,
                               auto_adjust=True, progress=False)

        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data["Close"]
        else:
            data = raw_data[["Close"]].rename(columns={"Close": tickers_to_download[0]})

    # Normalisation
    if normalize:
        data_plot = data / data.iloc[0] * 100
        ylabel = "Performance (base 100)"
    else:
        data_plot = data
        ylabel = "Prix ($)"

    # --- Métriques clés en haut ---
    st.markdown("### 📊 Vue d'ensemble")

    # Calcul des nouvelles métriques globales
    all_performances = []
    all_volatilities = []
    best_asset = None
    best_perf = float('-inf')

    for ticker in tickers:
        perf = ((data[ticker].iloc[-1] / data[ticker].iloc[0]) - 1) * 100
        returns = data[ticker].pct_change().dropna()
        vol = returns.std() * 100 * (252 ** 0.5)

        all_performances.append(perf)
        all_volatilities.append(vol)

        if perf > best_perf:
            best_perf = perf
            best_asset = ticker

    avg_perf = sum(all_performances) / len(all_performances) if all_performances else 0
    avg_vol = sum(all_volatilities) / len(all_volatilities) if all_volatilities else 0
    nb_assets = len(tickers)

    # Affichage des métriques globales
    overview_cols = st.columns(4)
    with overview_cols[0]:
        st.metric(
            label="📊 Performance moyenne",
            value=f"{avg_perf:+.2f}%",
            delta_color="normal" if avg_perf >= 0 else "inverse"
        )
    with overview_cols[1]:
        st.metric(
            label="🏆 Meilleur actif",
            value=best_asset if best_asset else "N/A",
            delta=f"{best_perf:+.2f}%"
        )
    with overview_cols[2]:
        st.metric(
            label="📈 Volatilité moyenne",
            value=f"{avg_vol:.2f}%"
        )
    with overview_cols[3]:
        st.metric(
            label="🔢 Actifs analysés",
            value=str(nb_assets)
        )

    st.markdown("---")

    # Métriques par actif
    st.markdown("#### Performance par actif")
    metrics_cols = st.columns(len(tickers_to_download))

    for idx, ticker in enumerate(tickers_to_download):
        with metrics_cols[idx]:
            perf = ((data[ticker].iloc[-1] / data[ticker].iloc[0]) - 1) * 100
            display_name = index_name if ticker == index_ticker else ticker
            delta_color = "normal" if perf >= 0 else "inverse"
            st.metric(
                label=display_name,
                value=f"${data[ticker].iloc[-1]:.2f}",
                delta=f"{perf:+.2f}%",
                delta_color=delta_color
            )

    # --- Graphique principal avec design moderne ---
    st.markdown("### 📈 Performance historique")

    data_plot_display = data_plot.copy()
    if compare_index and index_ticker in data_plot_display.columns:
        data_plot_display = data_plot_display.rename(columns={index_ticker: index_name})

    fig = go.Figure()

    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a']

    for idx, col in enumerate(data_plot_display.columns):
        is_index = (compare_index and col == index_name)
        fig.add_trace(go.Scatter(
            x=data_plot_display.index,
            y=data_plot_display[col],
            mode='lines',
            name=col,
            line=dict(
                color=colors[idx % len(colors)],
                width=3 if is_index else 2,
                dash='dash' if is_index else 'solid'
            ),
            hovertemplate='<b>%{fullData.name}</b><br>%{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        template='plotly_dark',
        hovermode='x unified',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        paper_bgcolor='rgba(15, 23, 42, 0)',
        font=dict(family='Arial', size=12, color='#e2e8f0'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='#667eea',
            borderwidth=1
        ),
        xaxis=dict(showgrid=True, gridcolor='rgba(102, 126, 234, 0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(102, 126, 234, 0.1)', title=ylabel),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- Graphiques secondaires ---
    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 💹 Rendements quotidiens")
        returns_data = data.pct_change().dropna() * 100
        if compare_index and index_ticker in returns_data.columns:
            returns_data = returns_data.rename(columns={index_ticker: index_name})

        fig_returns = go.Figure()
        for idx, col in enumerate(returns_data.columns):
            fig_returns.add_trace(go.Scatter(
                x=returns_data.index, y=returns_data[col], mode='lines',
                name=col, line=dict(color=colors[idx % len(colors)], width=1.5)
            ))

        fig_returns.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            paper_bgcolor='rgba(15, 23, 42, 0)',
            font=dict(color='#e2e8f0'),
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(102, 126, 234, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(102, 126, 234, 0.1)', title='Rendement (%)'),
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_returns, use_container_width=True)

    with col_right:
        st.markdown("### 📊 Volatilité glissante (30j)")
        rolling_vol = data.pct_change().rolling(window=30).std() * 100
        if compare_index and index_ticker in rolling_vol.columns:
            rolling_vol = rolling_vol.rename(columns={index_ticker: index_name})

        fig_vol = go.Figure()
        for idx, col in enumerate(rolling_vol.columns):
            fig_vol.add_trace(go.Scatter(
                x=rolling_vol.index, y=rolling_vol[col], mode='lines',
                name=col, line=dict(color=colors[idx % len(colors)], width=1.5),
                fill='tonexty' if idx > 0 else 'tozeroy',
                fillcolor=f'rgba({int(colors[idx % len(colors)][1:3], 16)}, {int(colors[idx % len(colors)][3:5], 16)}, {int(colors[idx % len(colors)][5:7], 16)}, 0.1)'
            ))

        fig_vol.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            paper_bgcolor='rgba(15, 23, 42, 0)',
            font=dict(color='#e2e8f0'),
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(102, 126, 234, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(102, 126, 234, 0.1)', title='Volatilité (%)'),
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # --- Stats détaillées ---
    if show_stats:
        st.markdown("---")
        st.markdown("### 📊 Analyse détaillée des performances")

        perf = pd.DataFrame()
        tickers_for_stats = tickers_to_download if compare_index else tickers

        for t in tickers_for_stats:
            returns = data[t].pct_change().dropna()
            running_max = data[t].cummax()
            drawdown = (data[t] / running_max - 1) * 100

            perf.loc[t, "Prix initial ($)"] = data[t].iloc[0]
            perf.loc[t, "Prix final ($)"] = data[t].iloc[-1]
            perf.loc[t, "Plus haut ($)"] = data[t].max()
            perf.loc[t, "Plus bas ($)"] = data[t].min()
            perf.loc[t, "Performance totale (%)"] = (data[t].iloc[-1] / data[t].iloc[0] - 1) * 100
            perf.loc[t, "Performance annualisée (%)"] = ((data[t].iloc[-1] / data[t].iloc[0]) ** (252 / len(data[t])) - 1) * 100
            perf.loc[t, "Volatilité quotidienne (%)"] = returns.std() * 100
            perf.loc[t, "Volatilité annualisée (%)"] = returns.std() * 100 * (252 ** 0.5)
            perf.loc[t, "Rendement moyen (%)"] = returns.mean() * 100
            perf.loc[t, "Rendement médian (%)"] = returns.median() * 100
            perf.loc[t, "Ratio Sharpe"] = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
            perf.loc[t, "Ratio Sortino"] = (returns.mean() / returns[returns < 0].std()) * (252 ** 0.5) if len(returns[returns < 0]) > 0 and returns[returns < 0].std() > 0 else 0
            perf.loc[t, "Drawdown max (%)"] = drawdown.min()
            perf.loc[t, "Drawdown actuel (%)"] = drawdown.iloc[-1]
            perf.loc[t, "Skewness"] = returns.skew()
            perf.loc[t, "Kurtosis"] = returns.kurtosis()

            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            perf.loc[t, "Jours positifs (%)"] = (len(positive_returns) / len(returns)) * 100
            perf.loc[t, "Gain moyen (%)"] = positive_returns.mean() * 100 if len(positive_returns) > 0 else 0
            perf.loc[t, "Perte moyenne (%)"] = negative_returns.mean() * 100 if len(negative_returns) > 0 else 0
            perf.loc[t, "Meilleur jour (%)"] = returns.max() * 100
            perf.loc[t, "Pire jour (%)"] = returns.min() * 100
            perf.loc[t, "Ratio Gain/Perte"] = abs(positive_returns.mean() / negative_returns.mean()) if len(negative_returns) > 0 and negative_returns.mean() != 0 else 0

        if compare_index and index_ticker in perf.index:
            perf = perf.rename(index={index_ticker: index_name})

        # Onglets avec dégradés de couleurs harmonieux
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Vue d'ensemble", "💰 Prix & Performance", "⚠️ Risque", "📊 Distribution"])

        with tab1:
            overview_cols = ["Performance totale (%)", "Performance annualisée (%)", "Volatilité annualisée (%)",
                             "Ratio Sharpe", "Drawdown max (%)", "Jours positifs (%)"]
            st.dataframe(
                perf[overview_cols].style.format("{:.2f}")
                .background_gradient(cmap="RdYlGn", subset=["Performance totale (%)", "Performance annualisée (%)"], axis=0)
                .background_gradient(cmap="YlOrRd", subset=["Volatilité annualisée (%)"], axis=0)
                .background_gradient(cmap="RdYlGn", subset=["Ratio Sharpe", "Jours positifs (%)"], axis=0)
                .background_gradient(cmap="RdYlGn_r", subset=["Drawdown max (%)"], axis=0),
                use_container_width=True
            )

        with tab2:
            price_cols = ["Prix initial ($)", "Prix final ($)", "Plus haut ($)", "Plus bas ($)",
                          "Performance totale (%)", "Performance annualisée (%)", "Rendement moyen (%)",
                          "Rendement médian (%)"]
            st.dataframe(
                perf[price_cols].style.format("{:.2f}")
                .background_gradient(cmap="Blues", subset=["Prix initial ($)", "Prix final ($)", "Plus haut ($)", "Plus bas ($)"], axis=0)
                .background_gradient(cmap="RdYlGn", subset=["Performance totale (%)", "Performance annualisée (%)", "Rendement moyen (%)", "Rendement médian (%)"], axis=0),
                use_container_width=True
            )

        with tab3:
            risk_cols = ["Volatilité quotidienne (%)", "Volatilité annualisée (%)", "Ratio Sharpe", "Ratio Sortino",
                         "Drawdown max (%)", "Drawdown actuel (%)", "Meilleur jour (%)", "Pire jour (%)"]
            st.dataframe(
                perf[risk_cols].style.format("{:.2f}")
                .background_gradient(cmap="YlOrRd", subset=["Volatilité quotidienne (%)", "Volatilité annualisée (%)"], axis=0)
                .background_gradient(cmap="RdYlGn", subset=["Ratio Sharpe", "Ratio Sortino"], axis=0)
                .background_gradient(cmap="RdYlGn_r", subset=["Drawdown max (%)", "Drawdown actuel (%)"], axis=0)
                .background_gradient(cmap="RdYlGn", subset=["Meilleur jour (%)"], axis=0)
                .background_gradient(cmap="RdYlGn_r", subset=["Pire jour (%)"], axis=0),
                use_container_width=True
            )

        with tab4:
            dist_cols = ["Jours positifs (%)", "Gain moyen (%)", "Perte moyenne (%)",
                         "Ratio Gain/Perte", "Skewness", "Kurtosis"]
            st.dataframe(
                perf[dist_cols].style.format("{:.2f}")
                .background_gradient(cmap="RdYlGn", subset=["Jours positifs (%)", "Gain moyen (%)", "Ratio Gain/Perte"], axis=0)
                .background_gradient(cmap="RdYlGn_r", subset=["Perte moyenne (%)"], axis=0)
                .background_gradient(cmap="RdBu_r", subset=["Skewness"], axis=0)
                .background_gradient(cmap="PuOr", subset=["Kurtosis"], axis=0),
                use_container_width=True
            )

        # --- Visualisations supplémentaires ---
        st.markdown("---")
        col_stat1, col_stat2 = st.columns(2)

        with col_stat1:
            st.markdown("### 🏆 Classement Performance")
            ranking = perf[["Performance totale (%)"]].sort_values("Performance totale (%)", ascending=False)
            st.dataframe(
                ranking.style.format("{:.2f}")
                .background_gradient(cmap="RdYlGn", axis=0),
                use_container_width=True
            )

        with col_stat2:
            st.markdown("### 🛡️ Classement Stabilité")
            risk_ranking = perf[["Volatilité annualisée (%)"]].sort_values("Volatilité annualisée (%)", ascending=True)
            st.dataframe(
                risk_ranking.style.format("{:.2f}")
                .background_gradient(cmap="RdYlGn", axis=0),
                use_container_width=True
            )

        # --- Ratio rendement/risque ---
        st.markdown("---")
        st.markdown("### ⚖️ Efficience (Rendement/Risque)")
        perf["Ratio Rend/Risque"] = perf["Performance totale (%)"] / perf["Volatilité annualisée (%)"]
        ratio_ranking = perf[["Performance totale (%)", "Volatilité annualisée (%)", "Ratio Rend/Risque"]].sort_values(
            "Ratio Rend/Risque", ascending=False)
        st.dataframe(
            ratio_ranking.style.format("{:.2f}")
            .background_gradient(cmap="RdYlGn", subset=["Performance totale (%)"], axis=0)
            .background_gradient(cmap="YlOrRd", subset=["Volatilité annualisée (%)"], axis=0)
            .background_gradient(cmap="RdYlGn", subset=["Ratio Rend/Risque"], axis=0),
            use_container_width=True
        )

        # --- Scatter plot moderne ---
        st.markdown("---")
        st.markdown("### 🎯 Matrice Risque-Rendement")
        scatter_data = perf.reset_index()
        scatter_data.columns = ["Ticker"] + list(scatter_data.columns[1:])

        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=scatter_data["Volatilité annualisée (%)"],
            y=scatter_data["Performance totale (%)"],
            mode='markers+text',
            text=scatter_data["Ticker"],
            textposition='top center',
            textfont=dict(size=12, color='white'),
            marker=dict(
                size=scatter_data["Performance totale (%)"].abs() + 10,
                color=scatter_data["Performance totale (%)"],
                colorscale='RdYlGn',
                showscale=True,
                line=dict(color='white', width=2),
                colorbar=dict(title="Perf %")
            ),
            hovertemplate='<b>%{text}</b><br>Risque: %{x:.2f}%<br>Rendement: %{y:.2f}%<extra></extra>'
        ))

        fig_scatter.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            paper_bgcolor='rgba(15, 23, 42, 0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(title='Risque (Volatilité %)', showgrid=True, gridcolor='rgba(102, 126, 234, 0.1)'),
            yaxis=dict(title='Rendement (%)', showgrid=True, gridcolor='rgba(102, 126, 234, 0.1)'),
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Téléchargement ---
    st.markdown("---")
    data_to_export = data.copy()
    if compare_index and index_ticker in data_to_export.columns:
        data_to_export = data_to_export.rename(columns={index_ticker: index_name})

    csv = data_to_export.to_csv().encode('utf-8')
    st.download_button(
        label="💾 Télécharger les données (CSV)",
        data=csv,
        file_name=f"moulachart_{source.replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Sélectionnez au moins un actif dans la barre latérale pour commencer l'analyse")