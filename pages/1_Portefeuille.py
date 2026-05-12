# --- Imports ---
import streamlit as st
import pandas as pd
import yfinance as yf
from yahooquery import search as yq_search
from datetime import datetime
from sqlalchemy import text
from streamlit_searchbox import st_searchbox
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration de la page ---
st.set_page_config(page_title="Mon Portefeuille", page_icon="💼", layout="wide")

# --- Authentification requise ---
if not st.user.is_logged_in:
    st.title("💼 Mon Portefeuille")
    st.warning("🔒 Vous devez être connecté pour accéder à votre portefeuille.")
    st.button("Se connecter avec Google", on_click=st.login)
    st.stop()

# --- Connexion à la base Neon (PostgreSQL) ---
conn = st.connection("postgresql", type="sql")

# --- Fonctions utilitaires ---
@st.cache_data(ttl=3600)
def validate_ticker(ticker: str) -> tuple[bool, str]:
    """Vérifie si un ticker est valide sur Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'regularMarketPrice' in info or 'currentPrice' in info:
            return True, info.get('longName', ticker)
        return False, None
    except:
        return False, None

@st.cache_data(ttl=3600)
def get_asset_details(ticker: str) -> dict:
    """Récupère les détails d'un actif (frais, dividendes, etc.)."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        expense_ratio = None
        if info.get('netExpenseRatio'):
            expense_ratio = info.get('netExpenseRatio') / 100
        elif info.get('expenseRatio'):
            expense_ratio = info.get('expenseRatio')
        elif info.get('annualReportExpenseRatio'):
            expense_ratio = info.get('annualReportExpenseRatio') / 100
        return {
            'expense_ratio': expense_ratio,
            'dividend_yield': info.get('dividendYield'),
            'dividend_rate': info.get('dividendRate'),
            'payout_ratio': info.get('payoutRatio'),
            'ex_dividend_date': info.get('exDividendDate'),
            'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield'),
            'trailing_annual_dividend_rate': info.get('trailingAnnualDividendRate'),
            'trailing_annual_dividend_yield': info.get('trailingAnnualDividendYield'),
            'category': info.get('category'),
            'fund_family': info.get('fundFamily'),
        }
    except:
        return {}

@st.cache_data(ttl=1800)
def download_portfolio_data(tickers_list: list) -> pd.DataFrame:
    """Télécharge les dernières données boursières pour une liste de tickers."""
    try:
        latest_prices = {}
        for ticker in tickers_list:
            try:
                ticker_data = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)
                if ticker_data.empty:
                    continue
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    ticker_data.columns = ticker_data.columns.get_level_values(0)
                if "Close" in ticker_data.columns:
                    close_series = ticker_data["Close"]
                elif len(ticker_data.columns) > 0:
                    close_series = ticker_data.iloc[:, 0]
                else:
                    continue
                clean_series = close_series.dropna()
                if len(clean_series) > 0:
                    latest_prices[ticker] = float(clean_series.iloc[-1])
            except Exception as e:
                st.warning(f"⚠️ Erreur pour {ticker}: {str(e)}")
                continue
        return pd.DataFrame([latest_prices], index=[pd.Timestamp.now()]) if latest_prices else None
    except Exception as e:
        st.warning(f"⚠️ Erreur lors du téléchargement des données : {str(e)}")
        return None

@st.cache_data(ttl=1800)
def calculate_volatility(tickers_list: list, period: str = "1y") -> dict:
    """Calcule la volatilité annualisée pour une liste de tickers."""
    vol_data = {}
    for ticker in tickers_list:
        try:
            df_ticker = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
            if df_ticker.empty:
                continue
            if isinstance(df_ticker.columns, pd.MultiIndex):
                df_ticker.columns = df_ticker.columns.get_level_values(0)
            prices = df_ticker["Close"].dropna()
            returns = prices.pct_change().dropna()
            vol_data[ticker] = returns.std() * (252 ** 0.5)
        except:
            vol_data[ticker] = None
    return vol_data

@st.cache_data(ttl=3600)
def get_asset_names(tickers_list: list) -> dict:
    """Récupère les noms complets des actifs pour une liste de tickers."""
    names = {}
    for ticker in tickers_list:
        try:
            info = yf.Ticker(ticker).info
            names[ticker] = info.get('longName', ticker)
        except:
            names[ticker] = ticker
    return names


@st.cache_data(ttl=3600)
def get_asset_sector_and_country(tickers_list: list) -> dict:
    """Récupère le secteur et le pays pour chaque ticker."""
    sectors = {}
    countries = {}
    for ticker in tickers_list:
        try:
            info = yf.Ticker(ticker).info
            sectors[ticker] = info.get('sector', 'Inconnu')
            countries[ticker] = info.get('country', 'Inconnu')
        except:
            sectors[ticker] = 'Inconnu'
            countries[ticker] = 'Inconnu'
    return {'sectors': sectors, 'countries': countries}

def search_assets_dynamic(search_term: str, **kwargs) -> list:
    """Fonction de recherche pour le composant searchbox."""
    if not search_term or len(search_term) < 2:
        return []
    try:
        results = yq_search(search_term)
        if not results or 'quotes' not in results:
            return []
        options = []
        for quote in results['quotes'][:10]:
            if quote.get('isYahooFinance', False):
                symbol = quote.get('symbol', '')
                name = quote.get('longname') or quote.get('shortname', '')
                exchange = quote.get('exchange', '')
                display = f"{symbol} - {name} ({exchange})"
                options.append(display)
        return options
    except:
        return []

def calculate_diversification_score(df_valid: pd.DataFrame, sectors: dict, countries: dict) -> tuple:
    """Calcule les scores de diversification sectorielle et géographique."""
    # --- Diversification sectorielle ---
    sector_counts = {}
    for ticker in df_valid['ticker']:
        sector = sectors.get(ticker, 'Inconnu')
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # Score sectoriel : 100% si tous les secteurs sont différents, 0% si tous les actifs sont dans le même secteur
    if len(sector_counts) == 0:
        sector_score = 0.0
    else:
        sector_score = (1 - max(sector_counts.values()) / len(df_valid)) * 100

    # --- Diversification géographique ---
    country_weights = {}
    for ticker in df_valid['ticker']:
        country = countries.get(ticker, 'Inconnu')
        weight = df_valid[df_valid['ticker'] == ticker]['Poids (%)'].iloc[0]
        country_weights[country] = country_weights.get(country, 0) + weight

    # Score géographique : 100% si les poids sont uniformément répartis, 0% si tout est concentré dans un seul pays
    if len(country_weights) == 0:
        geo_score = 0.0
    else:
        max_weight = max(country_weights.values())
        geo_score = (1 - max_weight / 100) * 100

    return sector_score, geo_score


# --- Barre latérale ---
with st.sidebar:
    st.title("💼 Mon Portefeuille")
    st.markdown("---")
    selected = st_searchbox(
        search_assets_dynamic,
        key="asset_searchbox",
        placeholder="nom, ticker, ISIN...",
        label="Recherche d'actif",
        clear_on_submit=False,
        clearable=True
    )
    ticker = selected.split(" - ")[0].strip() if selected else ""
    if ticker:
        st.info(f"**{selected}**")
    quantity = st.number_input("Quantité", min_value=0.0, step=1.0, format="%.4f", key="sidebar_qty")
    pru = st.number_input("PRU ($)", min_value=0.0, step=0.01, format="%.2f", key="sidebar_pru")
    if st.button("Ajouter au portefeuille", type="primary", width='stretch'):
        if not ticker:
            st.error("❌ Veuillez sélectionner un actif à partir du champ de recherche")
        elif quantity <= 0:
            st.error("❌ La quantité doit être supérieure à 0")
        else:
            with st.spinner(f"🔍 Vérification de {ticker}..."):
                is_valid, name = validate_ticker(ticker)
            if not is_valid:
                st.error(f"❌ Le ticker '{ticker}' n'est pas valide")
            else:
                try:
                    existing = conn.query(
                        "SELECT * FROM portfolio WHERE user_email = :email AND ticker = :ticker",
                        params={"email": st.user.email, "ticker": ticker},
                        ttl=0
                    )
                    if not existing.empty:
                        st.warning(f"⚠️ {ticker} existe déjà dans votre portefeuille.")
                    else:
                        with conn.session as session:
                            session.execute(
                                text("INSERT INTO portfolio (user_email, ticker, quantity, pru, date_added) "
                                     "VALUES (:email, :ticker, :qty, :pru, :date)"),
                                {
                                    "email": st.user.email,
                                    "ticker": ticker,
                                    "qty": quantity,
                                    "pru": pru,
                                    "date": datetime.now().strftime("%Y-%m-%d"),
                                }
                            )
                            session.commit()
                        st.success(f"✅ {ticker} ({name}) ajouté !")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
    st.markdown("---")
    st.sidebar.success(f"Connecté en tant que {st.user.name or st.user.email}")
    st.sidebar.button("Se déconnecter", on_click=st.logout, width='stretch')
    if st.sidebar.button("🔄 Actualiser les données", width='stretch'):
        st.cache_data.clear()
        st.rerun()

# --- Lecture du portefeuille utilisateur ---
try:
    df = conn.query("SELECT * FROM portfolio WHERE user_email = :email", params={"email": st.user.email}, ttl=0)
except Exception as e:
    st.error(f"❌ Erreur lors de la lecture du portefeuille : {str(e)}")
    st.stop()

if df.empty:
    st.info("🪙 Aucun actif enregistré. Ajoutez-en un ci-dessus.")
    st.stop()

# --- Téléchargement des données boursières ---
tickers = df["ticker"].unique().tolist()
with st.spinner("Chargement des cours boursiers..."):
    data = download_portfolio_data(tickers)

if data is None or data.empty:
    st.error("❌ Impossible de récupérer les données boursières")
    st.stop()

# --- Calcul des métriques ---
df["Dernier prix"] = None
for t in df["ticker"]:
    if t in data.columns:
        df.loc[df["ticker"] == t, "Dernier prix"] = data[t].iloc[-1]

df_valid = df[df["Dernier prix"].notna()].copy()
df_invalid = df[df["Dernier prix"].isna()].copy()

if not df_invalid.empty:
    st.warning(f"⚠️ Impossible de récupérer les prix pour : {', '.join(df_invalid['ticker'].tolist())}")

if df_valid.empty:
    st.error("❌ Aucun actif valide dans votre portefeuille")
    st.stop()

df_valid["Valeur actuelle"] = df_valid["quantity"] * df_valid["Dernier prix"]
df_valid["Investi"] = df_valid["quantity"] * df_valid["pru"]
df_valid["Gain/Perte ($)"] = df_valid["Valeur actuelle"] - df_valid["Investi"]
df_valid["Gain/Perte (%)"] = (df_valid["Valeur actuelle"] / df_valid["Investi"] - 1) * 100
df_valid["Poids (%)"] = (df_valid["Valeur actuelle"] / df_valid["Valeur actuelle"].sum()) * 100

# --- Ajouter le nom complet des actifs ---
names_dict = get_asset_names(df_valid["ticker"].tolist())
df_valid["Nom"] = df_valid["ticker"].apply(lambda x: names_dict.get(x, x))

# --- Calcul de la volatilité ---
vol_dict = calculate_volatility(df_valid["ticker"].tolist())
df_valid["Volatilité (%)"] = df_valid["ticker"].apply(lambda x: vol_dict.get(x) * 100 if vol_dict.get(x) else None)

# --- Résumé du portefeuille ---
st.subheader("Résumé global")

# Récupère les secteurs et pays
sector_country_info = get_asset_sector_and_country(df_valid["ticker"].tolist())
sectors = sector_country_info['sectors']
countries = sector_country_info['countries']

# Calcule les scores de diversification
sector_score, geo_score = calculate_diversification_score(df_valid, sectors, countries)

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
total_valeur = df_valid['Valeur actuelle'].sum()
total_investi = df_valid['Investi'].sum()
total_gain = df_valid['Gain/Perte ($)'].sum()
perf_totale = ((total_valeur / total_investi - 1) * 100) if total_investi > 0 else 0

# Calcul de la volatilité moyenne pondérée
volatilites = df_valid[['Poids (%)', 'Volatilité (%)']].dropna()
if not volatilites.empty:
    volatilite_moyenne = (volatilites['Poids (%)'] * volatilites['Volatilité (%)'] / 100).sum()
else:
    volatilite_moyenne = 0.0

with col1:
    st.metric("Valeur totale", f"${total_valeur:,.2f}", f"{perf_totale:+.2f}%")
with col2:
    st.metric("Capital investi", f"${total_investi:,.2f}")
with col3:
    st.metric("Gain/Perte", f"${total_gain:+,.2f}", delta_color="normal" if total_gain >= 0 else "inverse")
with col4:
    best_performer = df_valid.loc[df_valid["Gain/Perte (%)"].idxmax()]
    st.metric("Meilleur actif", best_performer["ticker"], f"{best_performer['Gain/Perte (%)']:+.2f}%")
with col5:
    st.metric("Volatilité moyenne", f"{volatilite_moyenne:.2f}%")
with col6:
    st.metric("Diversification sectorielle", f"{sector_score:.1f}%")
with col7:
    st.metric("Diversification géographique", f"{geo_score:.1f}%")


st.markdown("---")

# --- Affichage des données ---
st.subheader("Détails des positions")
df_display = df_valid.sort_values("Valeur actuelle", ascending=False)

def color_positive_negative(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return 'color: #4ade80'  # Vert clair
        elif val < 0:
            return 'color: #fb7185'  # Rouge clair
    return 'color: white'

st.dataframe(
    df_display[["ticker", "Nom", "quantity", "pru", "Dernier prix", "Valeur actuelle", "Poids (%)",
                "Gain/Perte ($)", "Gain/Perte (%)", "Volatilité (%)"]]
    .style.format({
        "quantity": "{:.4f}",
        "pru": "${:.2f}",
        "Dernier prix": "${:.2f}",
        "Valeur actuelle": "${:,.2f}",
        "Poids (%)": "{:.1f}%",
        "Gain/Perte ($)": "${:+,.2f}",
        "Gain/Perte (%)": "{:+.2f}%",
        "Volatilité (%)": "{:.2f}%"
    })
    .map(color_positive_negative, subset=["Gain/Perte (%)", "Gain/Perte ($)"]),
    width='stretch',
    hide_index=True
)

# --- Graphiques ---
st.markdown("---")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Répartition du portefeuille")
    fig_pie = px.pie(df_display, values="Valeur actuelle", names="ticker", hole=0.4,
                     color_discrete_sequence=px.colors.sequential.RdBu)
    fig_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(15, 23, 42, 0)', font=dict(color='#e2e8f0'),
                          showlegend=True)
    st.plotly_chart(fig_pie, config={'responsive': True})

with col_chart2:
    st.subheader("Performance par actif")
    fig_bar = go.Figure()
    colors_bar = ['#10b981' if x >= 0 else '#ef4444' for x in df_display["Gain/Perte (%)"]]
    fig_bar.add_trace(go.Bar(
        x=df_display["ticker"],
        y=df_display["Gain/Perte (%)"],
        marker_color=colors_bar,
        text=df_display["Gain/Perte (%)"].apply(lambda x: f"{x:+.1f}%"),
        textposition='outside'
    ))
    fig_bar.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(15, 23, 42, 0)',
        plot_bgcolor='rgba(15, 23, 42, 0.5)',
        font=dict(color='#e2e8f0'),
        showlegend=False,
        yaxis_title="Performance (%)",
        xaxis_title="Ticker",
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_bar, config={'responsive': True})

# --- Scanner de Frais et Dividendes ---
st.markdown("---")
st.subheader("Scanner de Frais & Dividendes")

with st.spinner("Analyse des frais et dividendes..."):
    fees_data = []
    for ticker in df_valid["ticker"].unique():
        details = get_asset_details(ticker)
        position = df_valid[df_valid["ticker"] == ticker].iloc[0]
        dividend_annual = None
        dividend_total = None
        if details.get('dividend_rate') and details['dividend_rate'] > 0:
            dividend_annual = details['dividend_rate']
            dividend_total = dividend_annual * position['quantity']
        elif details.get('trailing_annual_dividend_rate') and details['trailing_annual_dividend_rate'] > 0:
            dividend_annual = details['trailing_annual_dividend_rate']
            dividend_total = dividend_annual * position['quantity']
        fees_annual = None
        if details.get('expense_ratio') and details['expense_ratio'] > 0:
            fees_annual = position['Valeur actuelle'] * details['expense_ratio']
        fees_data.append({
            'Ticker': ticker,
            'Valeur': position['Valeur actuelle'],
            'Frais de gestion (%)': details.get('expense_ratio'),
            'Frais annuels ($)': fees_annual,
            'Rendement dividende (%)': details.get('dividend_yield') or details.get('trailing_annual_dividend_yield'),
            'Dividende annuel/action ($)': dividend_annual,
            'Dividendes annuels totaux ($)': dividend_total,
            'Taux de distribution (%)': details.get('payout_ratio'),
        })

df_fees = pd.DataFrame(fees_data)

# --- Onglets pour Frais et Dividendes ---
tab_fees, tab_div = st.tabs(["Frais de Gestion", "Dividendes"])

with tab_fees:
    df_with_fees = df_fees[df_fees['Frais de gestion (%)'].notna()].copy()
    if df_with_fees.empty:
        st.info("ℹ️ Aucun frais de gestion détecté (normal pour les actions individuelles)")
    else:
        col_f1, col_f2, col_f3 = st.columns(3)
        total_fees = df_with_fees['Frais annuels ($)'].sum()
        avg_expense_ratio = (df_with_fees['Frais de gestion (%)'] * 100).mean()
        with col_f1:
            st.metric("Frais annuels totaux", f"${total_fees:,.2f}")
        with col_f2:
            st.metric("Ratio moyen", f"{avg_expense_ratio:.3f}%")
        with col_f3:
            highest_fee = df_with_fees.loc[df_with_fees['Frais annuels ($)'].idxmax()]
            st.metric("Plus élevé", highest_fee['Ticker'], f"${highest_fee['Frais annuels ($)']:,.2f}")


        def color_fees(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: #fb7185'  # Rouge clair
            return 'color: white'


        st.dataframe(
            df_with_fees[['Ticker', 'Valeur', 'Frais de gestion (%)', 'Frais annuels ($)']]
            .style.format({
                'Valeur': '${:,.2f}',
                'Frais de gestion (%)': '{:.3%}',
                'Frais annuels ($)': '${:,.2f}'
            })
            .map(color_fees, subset=['Frais annuels ($)']),
            width='stretch',
            hide_index=True
        )

# --- Récapitulatif des secteurs et pays ---
st.markdown("---")
st.subheader("Diversification")

# Secteurs
sector_weights = {}
for ticker in df_valid['ticker']:
    sector = sectors.get(ticker, 'Inconnu')
    weight = df_valid[df_valid['ticker'] == ticker]['Poids (%)'].iloc[0]
    sector_weights[sector] = sector_weights.get(sector, 0) + weight

# Pays
country_weights = {}
for ticker in df_valid['ticker']:
    country = countries.get(ticker, 'Inconnu')
    weight = df_valid[df_valid['ticker'] == ticker]['Poids (%)'].iloc[0]
    country_weights[country] = country_weights.get(country, 0) + weight

col_sector, col_country = st.columns(2)

with col_sector:
    st.write("### Répartition sectorielle")
    sector_df = pd.DataFrame.from_dict(sector_weights, orient='index', columns=['Poids (%)'])
    st.dataframe(
        sector_df.style.format({"Poids (%)": "{:.1f}%"}),
        width='stretch',
        hide_index=False
    )

with col_country:
    st.write("### Répartition géographique")
    country_df = pd.DataFrame.from_dict(country_weights, orient='index', columns=['Poids (%)'])
    st.dataframe(
        country_df.style.format({"Poids (%)": "{:.1f}%"}),
        width='stretch',
        hide_index=False
    )


with tab_div:
    df_with_div = df_fees[df_fees['Dividendes annuels totaux ($)'].notna()].copy()
    if df_with_div.empty:
        st.info("ℹ️ Aucun dividende détecté dans votre portefeuille")
    else:
        col_d1, col_d2, col_d3 = st.columns(3)
        total_dividends = df_with_div['Dividendes annuels totaux ($)'].sum()
        avg_yield = (df_with_div['Rendement dividende (%)'] * 100).mean()
        with col_d1:
            st.metric("Dividendes annuels totaux", f"${total_dividends:,.2f}")
        with col_d2:
            st.metric("Rendement moyen", f"{avg_yield:.2f}%")
        with col_d3:
            st.metric("Revenus mensuels estimés", f"${total_dividends / 12:,.2f}")


        def color_dividends(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: #4ade80'  # Vert clair
            return 'color: white'


        st.dataframe(
            df_with_div[['Ticker', 'Valeur', 'Rendement dividende (%)', 'Dividende annuel/action ($)',
                         'Dividendes annuels totaux ($)', 'Taux de distribution (%)']]
            .style.format({
                'Valeur': '${:,.2f}',
                'Rendement dividende (%)': '{:.2%}',
                'Dividende annuel/action ($)': '${:.2f}',
                'Dividendes annuels totaux ($)': '${:.2f}',
                'Taux de distribution (%)': '{:.2%}'
            })
            .map(color_dividends, subset=['Dividendes annuels totaux ($)', 'Rendement dividende (%)']),
            width='stretch',
            hide_index=True
        )

# --- Actions rapides ---
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Modifier", "Supprimer", "Exporter"])

with tab1:
    edit_ticker = st.selectbox("Actif", options=df["ticker"].tolist(), key="edit")
    if edit_ticker:
        current = df[df["ticker"] == edit_ticker].iloc[0]
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            new_qty = st.number_input("Quantité", value=float(current["quantity"]), min_value=0.0, step=1.0,
                                      format="%.4f")
        with col_e2:
            new_pru = st.number_input("PRU ($)", value=float(current["pru"]), min_value=0.0, step=0.01, format="%.2f")
        if st.button("Mettre à jour", type="primary",  width='stretch'):
            if new_qty <= 0 or new_pru <= 0:
                st.error("Valeurs invalides")
            else:
                try:
                    with conn.session as session:
                        session.execute(
                            text("UPDATE portfolio SET quantity = :qty, pru = :pru "
                                 "WHERE user_email = :email AND ticker = :ticker"),
                            {"qty": new_qty, "pru": new_pru, "email": st.user.email, "ticker": edit_ticker}
                        )
                        session.commit()
                    st.success(f"✓ {edit_ticker} mis à jour")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {str(e)}")

with tab2:
    delete_ticker = st.selectbox("Actif", options=df["ticker"].tolist(), key="delete")
    if delete_ticker:
        info = df[df["ticker"] == delete_ticker].iloc[0]
        st.warning(f"⚠️ Supprimer {delete_ticker} ({info['quantity']} unités)")
        if st.button("Confirmer", type="primary",  width='stretch'):
            try:
                with conn.session as session:
                    session.execute(
                        text("DELETE FROM portfolio WHERE user_email = :email AND ticker = :ticker"),
                        {"email": st.user.email, "ticker": delete_ticker}
                    )
                    session.commit()
                st.success(f"✓ {delete_ticker} supprimé")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {str(e)}")

with tab3:
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        csv_export = df_display[
            ["ticker", "quantity", "pru", "Dernier prix", "Valeur actuelle", "Gain/Perte ($)",
             "Gain/Perte (%)"]].to_csv(
            index=False).encode('utf-8')
        st.download_button(
            label="Télécharger le portefeuille (CSV)",
            data=csv_export,
            file_name=f"portefeuille_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
             width='stretch'
        )
    with col_export2:
        total_fees_report = df_fees[df_fees['Frais annuels ($)'].notna()]['Frais annuels ($)'].sum()
        total_div_report = df_fees[df_fees['Dividendes annuels totaux ($)'].notna()][
            'Dividendes annuels totaux ($)'].sum()
        rapport = f"""RAPPORT DE PORTEFEUILLE
Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Utilisateur: {st.user.email}
RÉSUMÉ
------
Valeur totale: ${total_valeur:,.2f}
Capital investi: ${total_investi:,.2f}
Gain/Perte: ${total_gain:+,.2f} ({perf_totale:+.2f}%)
Nombre d'actifs: {len(df_valid)}
FRAIS & DIVIDENDES
------------------
Frais annuels totaux: ${total_fees_report:,.2f}
Dividendes annuels totaux: ${total_div_report:,.2f}
Revenus nets estimés: ${total_div_report - total_fees_report:,.2f}
POSITIONS
---------
{df_display[['ticker', 'quantity', 'Dernier prix', 'Valeur actuelle', 'Gain/Perte (%)']].to_string()}
"""
        st.download_button(
            label="Télécharger le rapport (TXT)",
            data=rapport.encode('utf-8'),
            file_name=f"rapport_portefeuille_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
             width='stretch'
        )
