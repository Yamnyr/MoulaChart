import streamlit as st
import pandas as pd
import yfinance as yf
from yahooquery import search as yq_search
from datetime import datetime
from supabase import create_client, Client
from streamlit_searchbox import st_searchbox

st.set_page_config(page_title="Mon Portefeuille", page_icon="💼", layout="wide")

# --- Authentification requise ---
if not st.user.is_logged_in:
    st.title("💼 Mon Portefeuille")
    st.warning("🔒 Vous devez être connecté pour accéder à votre portefeuille.")
    st.button("Se connecter avec Google", on_click=st.login)
    st.stop()

# --- Barre latérale ---
# st.sidebar.success(f"Connecté en tant que {st.user.name or st.user.email}")
# st.sidebar.button("🚪 Se déconnecter", on_click=st.logout)


# --- Connexion à la base Supabase ---
@st.cache_resource
def init_supabase() -> Client:
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        st.error(f"❌ Erreur de connexion à Supabase : {str(e)}")
        st.stop()


supabase = init_supabase()


# --- Fonction de recherche pour searchbox ---
def search_assets_dynamic(search_term: str, **kwargs):
    """Fonction de recherche pour le composant searchbox"""
    if not search_term or len(search_term) < 2:
        return []

    try:
        results = yq_search(search_term)

        if not results or 'quotes' not in results:
            return []

        # Retourner une liste de strings
        options = []
        for quote in results['quotes'][:10]:
            if quote.get('isYahooFinance', False):
                symbol = quote.get('symbol', '')
                name = quote.get('longname') or quote.get('shortname', '')
                exchange = quote.get('exchange', '')
                display = f"{symbol} - {name} ({exchange})"
                options.append(display)

        return options
    except Exception as e:
        return []


# --- Validation d'un ticker ---
@st.cache_data(ttl=3600)
def validate_ticker(ticker):
    """Vérifie si un ticker existe sur Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if 'regularMarketPrice' in info or 'currentPrice' in info:
            return True, info.get('longName', ticker)
        return False, None
    except:
        return False, None


# --- Téléchargement des données avec cache ---# --- Téléchargement des données avec cache ---
@st.cache_data(ttl=1800)
def download_portfolio_data(tickers_list):
    """Télécharge les données pour le portefeuille"""
    try:
        if not tickers_list:
            return None

        # Dictionnaire pour stocker le dernier prix de chaque ticker
        latest_prices = {}

        for ticker in tickers_list:
            try:
                # Télécharger les données
                ticker_data = yf.download(ticker, period="5d", interval="1d", auto_adjust=True, progress=False)

                if ticker_data.empty:
                    continue

                # Gérer le cas MultiIndex (quand yfinance retourne plusieurs niveaux de colonnes)
                if isinstance(ticker_data.columns, pd.MultiIndex):
                    # Aplatir les colonnes MultiIndex
                    ticker_data.columns = ticker_data.columns.get_level_values(0)

                # Extraire la colonne Close
                if "Close" in ticker_data.columns:
                    close_series = ticker_data["Close"]
                elif len(ticker_data.columns) > 0:
                    # Prendre la première colonne disponible
                    close_series = ticker_data.iloc[:, 0]
                else:
                    continue

                # Récupérer le dernier prix valide (non-NaN)
                clean_series = close_series.dropna()
                if len(clean_series) > 0:
                    last_price = float(clean_series.iloc[-1])
                    latest_prices[ticker] = last_price

            except Exception as e:
                st.warning(f"⚠️ Erreur pour {ticker}: {str(e)}")
                continue

        if not latest_prices:
            return None

        # Créer un DataFrame simple avec une seule ligne contenant les derniers prix
        data = pd.DataFrame([latest_prices], index=[pd.Timestamp.now()])
        return data

    except Exception as e:
        st.warning(f"⚠️ Erreur lors du téléchargement des données : {str(e)}")
        return None


# st.title("💼 Mon Portefeuille")
# st.caption("Gérez vos actifs et suivez leurs performances.")

# --- Formulaire d'ajout dans la barre latérale ---
with st.sidebar:
    st.title("💼 Mon Portefeuille")
    # st.subheader("Ajouter un actif")
    st.markdown("---")

    # st.subheader("Ajouter un actif")

    # Champ de recherche
    selected = st_searchbox(
        search_assets_dynamic,
        key="asset_searchbox",
        placeholder="nom, ticker, ISIN...",
        label="Recherche d'actif",
        clear_on_submit=False,
        clearable=True
    )

    ticker = ""
    if selected:
        ticker = selected.split(" - ")[0].strip()

    if ticker:
        st.info(f"**{selected}**")

    quantity = st.number_input("Quantité", min_value=0.0, step=1.0, format="%.4f", key="sidebar_qty")
    pru = st.number_input("PRU ($)", min_value=0.0, step=0.01, format="%.2f", key="sidebar_pru")

    if st.button("Ajouter au portefeuille", type="primary", use_container_width=True):
        if not ticker:
            st.error("❌ Veuillez sélectionner un actif à partir du champ de recherche")
        elif quantity <= 0:
            st.error("❌ La quantité doit être supérieure à 0")
        # elif pru <= 0:
        #     st.error("❌ Le PRU doit être supérieur à 0")
        else:
            with st.spinner(f"🔍 Vérification de {ticker}..."):
                is_valid, name = validate_ticker(ticker)

            if not is_valid:
                st.error(f"❌ Le ticker '{ticker}' n'est pas valide")
            else:
                try:
                    existing = supabase.table("portfolio").select("*") \
                        .eq("user_email", st.user.email).eq("ticker", ticker).execute()

                    if existing.data:
                        st.warning(f"⚠️ {ticker} existe déjà dans votre portefeuille.")
                    else:
                        data = {
                            "user_email": st.user.email,
                            "ticker": ticker,
                            "quantity": quantity,
                            "pru": pru,
                            "date_added": datetime.now().strftime("%Y-%m-%d"),
                        }
                        supabase.table("portfolio").insert(data).execute()
                        st.success(f"✅ {ticker} ({name}) ajouté !")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")

    st.markdown("---")
    st.sidebar.success(f"Connecté en tant que {st.user.name or st.user.email}")
    st.sidebar.button("Se déconnecter", on_click=st.logout,  width='stretch')


# --- Lecture du portefeuille utilisateur ---
try:
    response = supabase.table("portfolio").select("*").eq("user_email", st.user.email).execute()
    df = pd.DataFrame(response.data)
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

# --- Résumé du portefeuille ---
st.subheader("Résumé global")

col1, col2, col3, col4 = st.columns(4)

total_valeur = df_valid['Valeur actuelle'].sum()
total_investi = df_valid['Investi'].sum()
total_gain = df_valid['Gain/Perte ($)'].sum()
perf_totale = ((total_valeur / total_investi - 1) * 100) if total_investi > 0 else 0

with col1:
    st.metric("Valeur totale", f"${total_valeur:,.2f}", f"{perf_totale:+.2f}%")

with col2:
    st.metric("Capital investi", f"${total_investi:,.2f}")

with col3:
    st.metric("Gain/Perte", f"${total_gain:+,.2f}", delta_color="normal" if total_gain >= 0 else "inverse")

with col4:
    best_performer = df_valid.loc[df_valid["Gain/Perte (%)"].idxmax()]
    st.metric("Meilleur actif", best_performer["ticker"], f"{best_performer['Gain/Perte (%)']:+.2f}%")

st.markdown("---")

# --- Affichage des données ---
st.subheader("Détails des positions")

df_display = df_valid.sort_values("Valeur actuelle", ascending=False)

st.dataframe(
    df_display[["ticker", "quantity", "pru", "Dernier prix", "Valeur actuelle", "Poids (%)", "Gain/Perte ($)",
                "Gain/Perte (%)"]]
    .style.format({
        "quantity": "{:.4f}",
        "pru": "${:.2f}",
        "Dernier prix": "${:.2f}",
        "Valeur actuelle": "${:,.2f}",
        "Poids (%)": "{:.1f}%",
        "Gain/Perte ($)": "${:+,.2f}",
        "Gain/Perte (%)": "{:+.2f}%"
    })
    .background_gradient(cmap="RdYlGn", subset=["Gain/Perte (%)"], axis=0)
    .background_gradient(cmap="Blues", subset=["Poids (%)"], axis=0),
     width='stretch',
    hide_index=True
)

# --- Graphiques ---
st.markdown("---")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Répartition du portefeuille")
    import plotly.express as px

    fig_pie = px.pie(df_display, values="Valeur actuelle", names="ticker", hole=0.4,
                     color_discrete_sequence=px.colors.sequential.RdBu)
    fig_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(15, 23, 42, 0)', font=dict(color='#e2e8f0'),
                          showlegend=True)
    st.plotly_chart(fig_pie, config={'responsive': True})

with col_chart2:
    st.subheader("Performance par actif")
    import plotly.graph_objects as go

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

# --- Modification et suppression ---
st.markdown("---")
# col_edit, col_delete = st.columns(2)

# --- Actions rapides ---
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
                    supabase.table("portfolio").update({"quantity": new_qty, "pru": new_pru}).eq("user_email",
                                                                                                 st.user.email).eq(
                        "ticker", edit_ticker).execute()
                    st.success(f"✓ {edit_ticker} mis à jour")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {str(e)}")

with tab2:
    delete_ticker = st.selectbox("Actif", options=df["ticker"].tolist(), key="delete")
    if delete_ticker:
        info = df[df["ticker"] == delete_ticker].iloc[0]
        st.warning(f"⚠️ Supprimer {delete_ticker} ({info['quantity']} unités)")
        # col_d1, col_d2 = st.columns(2)
        # with col_d1:
        if st.button("Confirmer", type="primary",  width='stretch'):
            try:
                supabase.table("portfolio").delete().eq("user_email", st.user.email).eq("ticker",
                                                                                        delete_ticker).execute()
                st.success(f"✓ {delete_ticker} supprimé")
                st.rerun()
            except Exception as e:
                st.error(f"Erreur : {str(e)}")

with tab3:
    # st.markdown("---")
    # st.subheader("💾 Exporter les données")

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
        rapport = f"""RAPPORT DE PORTEFEUILLE
    Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}
    Utilisateur: {st.user.email}

    RÉSUMÉ
    ------
    Valeur totale: ${total_valeur:,.2f}
    Capital investi: ${total_investi:,.2f}
    Gain/Perte: ${total_gain:+,.2f} ({perf_totale:+.2f}%)
    Nombre d'actifs: {len(df_valid)}

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

# --- Export ---
