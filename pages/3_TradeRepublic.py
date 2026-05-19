import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# ── MCC categories (Clean plain text) ───────────────────────────
MCC_CATEGORIES = {
    "4784": "Peages", "4131": "Transport", "5541": "Carburant",
    "5814": "Restauration rapide", "5812": "Restaurants", "5411": "Alimentation",
    "5542": "Alimentation", "5331": "Alimentation", "5499": "Alimentation",
    "7230": "Beaute", "8011": "Sante", "5912": "Sante",
    "5631": "Habillement", "5651": "Habillement",
    "5941": "Sport", "5734": "Loisirs", "7996": "Loisirs",
    "5999": "Divers", "5965": "E-commerce",
    "7399": "Services", "9405": "Impots/Admin", "5462": "Boulangerie",
}

# ── Load data ────────────────────────────────────────────────────
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, parse_dates=["date", "datetime"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0)
    df["tax"] = pd.to_numeric(df["tax"], errors="coerce").fillna(0)
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    
    # Fix: pandas read numeric MCCs as float (5814.0) -> remove .0 to match MCC_CATEGORIES dict
    df["mcc_code"] = (df["mcc_code"].astype(str).str.strip()
                      .str.replace(r"\.0$", "", regex=True)
                      .replace("nan", "").replace("", pd.NA))
    df["category_label"] = df["mcc_code"].map(MCC_CATEGORIES).fillna("Divers")
    
    # Rename DEFAULT -> CTO
    df["account_type"] = df["account_type"].replace("DEFAULT", "CTO")
    
    # Dynamic ETF Name Parsing: Use 'name' column if populated, fallback to symbol/ISIN
    df["etf_name"] = df["name"].fillna(df["symbol"]).fillna("Inconnu")
    
    # Clean ETF names to look beautiful in charts
    df["etf_name"] = df["etf_name"].astype(str).str.replace(r"\s*(USD|EUR)?\s*\(Acc\)\s*", "", regex=True)
    df["etf_name"] = df["etf_name"].str.replace(r"\s*Swap\s*", " ", regex=True)
    df["etf_name"] = df["etf_name"].str.replace(r"\s*UCITS ETF\s*", " ", regex=True)
    df["etf_name"] = df["etf_name"].str.replace(r"iShares VI plc - iShares\s*", "iShares ", regex=True)
    df["etf_name"] = df["etf_name"].str.replace(r"iShares VII plc - iShares\s*", "iShares ", regex=True)
    df["etf_name"] = df["etf_name"].str.strip()
    
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("Trade Republic")
    st.markdown("---")
    
    # Toggle to hide amounts
    hide_amounts = st.toggle("Masquer les montants", value=False, help="Masque les valeurs numériques sur l'ensemble de la page")
    st.markdown("---")
    
    default_path = os.path.join(os.path.dirname(__file__), "..", "Exportation de transactions.csv")
    if os.path.exists(default_path):
        st.success("Fichier TR detecte automatiquement")
        uploaded = None
    else:
        uploaded = st.file_uploader("Importer votre export TR (CSV)", type=["csv"])
    st.markdown("---")
    account_filter = st.multiselect("Compte", ["CTO", "PEA"], default=["CTO", "PEA"])

# ── Formatting Helper ────────────────────────────────────────────
def fmt(val, is_curr=True, decimals=2):
    if hide_amounts:
        return "•••• €" if is_curr else "••••"
    if is_curr:
        return f"{val:,.{decimals}f} €"
    return f"{val:,.{decimals}f}"

def mask_df(df_to_mask, columns_to_mask, is_curr=True):
    if hide_amounts:
        copy_df = df_to_mask.copy()
        for col in columns_to_mask:
            if col in copy_df.columns:
                copy_df[col] = "•••• €" if is_curr else "••••"
        return copy_df
    return df_to_mask

# ── Load ─────────────────────────────────────────────────────────
try:
    if uploaded:
        df = load_csv(uploaded)
    elif os.path.exists(default_path):
        df = load_csv(default_path)
    else:
        st.info("Importez votre export Trade Republic pour commencer.")
        st.stop()
except Exception as e:
    st.error(f"Erreur lecture CSV : {e}")
    st.stop()

df = df[df["account_type"].isin(account_filter)]

# Colors setup
ACCOUNT_COLORS = {"CTO": "#10b981", "PEA": "#3b82f6"}

# Dynamic ETF Colors Mapping
unique_etfs = sorted(df["etf_name"].dropna().unique())
color_palette = ["#10b981", "#3b82f6", "#a855f7", "#f97316", "#ec4899", "#f59e0b", "#14b8a6", "#6366f1"]
etf_colors = {name: color_palette[i % len(color_palette)] for i, name in enumerate(unique_etfs)}

# ── Subsets ──────────────────────────────────────────────────────
buys    = df[(df["type"] == "BUY") & (df["category"] == "TRADING")]
cards   = df[(df["type"] == "CARD_TRANSACTION") & (df["amount"] < 0)]
inflows = df[df["type"].isin(["CUSTOMER_INBOUND", "TRANSFER_INSTANT_INBOUND"])]
savings = df[df["type"] == "BENEFITS_SAVEBACK"]
interests = df[df["type"] == "INTEREST_PAYMENT"]

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.metric-card{background:linear-gradient(135deg,#1a1b1e,#2d2d30);border:1px solid #3d3d40;
border-radius:12px;padding:18px 22px;margin-bottom:8px;}
.metric-label{font-size:12px;color:#94a3b8;font-weight:500;text-transform:uppercase;letter-spacing:.5px;}
.metric-value{font-size:26px;font-weight:700;color:#f1f5f9;margin-top:4px;}
.metric-delta-pos{font-size:13px;color:#10b981;margin-top:2px;}
.metric-delta-neg{font-size:13px;color:#ef4444;margin-top:2px;}
.section-title{font-size:22px;font-weight:700;color:#f1f5f9;margin:28px 0 16px;
border-left:4px solid #10b981;padding-left:12px;}
.tax-card{background:#111827;border:1px solid #374151;border-radius:10px;padding:15px;margin-bottom:15px;}
.tax-box{background:#1f2937;border:1px dashed #4b5563;border-radius:6px;padding:8px 12px;margin:5px 0;}
.tax-label{font-size:11px;color:#9ca3af;text-transform:uppercase;}
.tax-value{font-size:16px;font-weight:bold;color:#f3f4f6;}
</style>
""", unsafe_allow_html=True)

st.markdown("# Analyse Trade Republic")
st.markdown(f"*{len(df)} transactions · {df['date'].min().strftime('%d/%m/%Y')} → {df['date'].max().strftime('%d/%m/%Y')}*")
st.markdown("---")

# ════════════════════════════════════════════════════════════════
# KPIs
# ════════════════════════════════════════════════════════════════
total_invested = buys["amount"].abs().sum()
total_deposited = inflows["amount"].sum()
total_saveback = savings["amount"].sum()
total_interest = interests["amount"].sum()
total_spent = cards["amount"].abs().sum()

c1, c2, c3, c4, c5 = st.columns(5)
def kpi(col, label, value, delta=None, pos=True):
    dclass = "metric-delta-pos" if pos else "metric-delta-neg"
    delta_html = f'<div class="{dclass}">{delta}</div>' if delta else ""
    col.markdown(f"""<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>{delta_html}</div>""", unsafe_allow_html=True)

kpi(c1, "Total investi", fmt(total_invested, decimals=0))
kpi(c2, "Depots totaux", fmt(total_deposited, decimals=0))
kpi(c3, "Saveback recu", fmt(total_saveback), "Cashback 1%")
kpi(c4, "Interets recus", fmt(total_interest))
kpi(c5, "Depenses carte", fmt(total_spent, decimals=0), f"{len(cards)} transactions", False)

# Main Navigation Tabs
main_tab1, main_tab2, main_tab3 = st.tabs(["Investissements", "Depenses & Budget", "Simulateur IFU"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — MES INVESTISSEMENTS
# ════════════════════════════════════════════════════════════════
with main_tab1:
    tab_inv1, tab_inv2, tab_inv3 = st.tabs(["Evolution", "Repartition", "Detail par ETF"])

    with tab_inv1:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            # Cumulative investment per ETF over time
            buys_sorted = buys.sort_values("date")
            cum_data = []
            for etf in buys_sorted["etf_name"].unique():
                sub = buys_sorted[buys_sorted["etf_name"] == etf].copy()
                sub["cum_invested"] = sub["amount"].abs().cumsum()
                sub["etf"] = etf
                cum_data.append(sub[["date", "cum_invested", "etf"]])

            if cum_data:
                df_cum = pd.concat(cum_data)
                fig = go.Figure()
                for etf in df_cum["etf"].unique():
                    s = df_cum[df_cum["etf"] == etf]
                    color = etf_colors.get(etf, "#64748b")
                    fig.add_trace(go.Scatter(
                        x=s["date"], y=s["cum_invested"], name=etf, mode="lines",
                        line=dict(color=color, width=2.5),
                        hovertemplate=f"<b>{etf}</b><br>%{{x|%d/%m/%Y}}<br>" + ("•••• €" if hide_amounts else "%{y:,.0f} €") + "<extra></extra>"
                    ))
                fig.update_layout(
                    title="Capital cumulé investi par ETF",
                    template="plotly_dark", hovermode="x unified",
                    plot_bgcolor="rgba(15,23,42,0.5)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"), height=400,
                    legend=dict(orientation="h", y=1.08, x=0),
                    xaxis=dict(showgrid=True, gridcolor="rgba(102,126,234,0.1)"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(102,126,234,0.1)", title="€ investi", showticklabels=not hide_amounts),
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            # Monthly investment bar
            monthly_buys = buys.groupby("month")["amount"].apply(lambda x: x.abs().sum()).reset_index()
            monthly_buys.columns = ["month", "montant"]
            fig2 = go.Figure(go.Bar(
                x=monthly_buys["month"], y=monthly_buys["montant"],
                marker_color="#10b981",
                hovertemplate="<b>%{x}</b><br>" + ("•••• €" if hide_amounts else "%{y:,.0f} €") + "<extra></extra>"
            ))
            fig2.update_layout(
                title="Investissement mensuel",
                template="plotly_dark", plot_bgcolor="rgba(15,23,42,0.5)",
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                height=400, xaxis=dict(showgrid=False, tickangle=45),
                yaxis=dict(showgrid=True, gridcolor="rgba(102,126,234,0.1)", title="€", showticklabels=not hide_amounts),
                margin=dict(l=0, r=0, t=50, b=60)
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab_inv2:
        col_a, col_b = st.columns(2)
        etf_totals = buys.groupby("etf_name")["amount"].apply(lambda x: x.abs().sum()).reset_index()
        etf_totals.columns = ["ETF", "Montant"]

        with col_a:
            colors_pie = [etf_colors.get(e, "#64748b") for e in etf_totals["ETF"]]
            fig_pie = go.Figure(go.Pie(
                labels=etf_totals["ETF"], values=etf_totals["Montant"],
                hole=0.55, marker=dict(colors=colors_pie),
                textinfo="label+percent" if not hide_amounts else "label",
                hovertemplate="<b>%{label}</b><br>" + ("•••• €" if hide_amounts else "%{value:,.0f} €<br>%{percent}") + "<extra></extra>"
            ))
            fig_pie.update_layout(
                title="Répartition du capital investi", template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
                height=380, showlegend=False,
                annotations=[dict(text=fmt(total_invested, decimals=0), x=0.5, y=0.5,
                                  font=dict(size=18, color="white"), showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            # Dynamic CTO vs PEA bar chart
            etf_account = buys.groupby(["etf_name", "account_type"])["amount"].apply(
                lambda x: x.abs().sum()).reset_index()
            etf_account.columns = ["ETF", "Compte", "Investi"]
            fig_cto_pea = go.Figure()
            for compte in etf_account["Compte"].unique():
                sub = etf_account[etf_account["Compte"] == compte]
                fig_cto_pea.add_trace(go.Bar(
                    x=sub["ETF"], y=sub["Investi"], name=compte,
                    marker_color=ACCOUNT_COLORS.get(compte, "#64748b"),
                    text=sub["Investi"].apply(lambda v: fmt(v, decimals=0)) if not hide_amounts else None,
                    textposition="auto",
                    hovertemplate="<b>%{x}</b> (" + compte + ")<br>" + ("•••• €" if hide_amounts else "%{y:,.0f} €") + "<extra></extra>"
                ))
            fig_cto_pea.update_layout(
                title="Capital investi par ETF — CTO vs PEA", template="plotly_dark",
                plot_bgcolor="rgba(15,23,42,0.5)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), height=380, barmode="group",
                legend=dict(orientation="h", y=1.08),
                yaxis=dict(showgrid=True, gridcolor="rgba(102,126,234,0.1)", title="€ investi", showticklabels=not hide_amounts),
                margin=dict(l=0, r=0, t=60, b=0)
            )
            st.plotly_chart(fig_cto_pea, use_container_width=True)

    with tab_inv3:
        # Totaux CTO vs PEA
        cto_total = buys[buys["account_type"] == "CTO"]["amount"].abs().sum()
        pea_total = buys[buys["account_type"] == "PEA"]["amount"].abs().sum()
        grand_total = cto_total + pea_total
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("CTO — Total investi", fmt(cto_total, decimals=0),
                   f"{cto_total/grand_total*100:.1f}%" if grand_total and not hide_amounts else "")
        mc2.metric("PEA — Total investi", fmt(pea_total, decimals=0),
                   f"{pea_total/grand_total*100:.1f}%" if grand_total and not hide_amounts else "")
        mc3.metric("Total tous comptes", fmt(grand_total, decimals=0))
        st.markdown("---")
        
        etf_detail = buys.groupby(["etf_name", "account_type"]).agg(
            Achats=("amount", "count"),
            Investi=("amount", lambda x: x.abs().sum()),
            Parts=("shares", "sum"),
            Prix_moyen=("price", "mean")
        ).reset_index()
        etf_detail.columns = ["ETF", "Compte", "Nb achats", "Investi (€)", "Parts totales", "Prix moy./part (€)"]
        etf_detail["Valeur/part actuelle (€)"] = etf_detail["Investi (€)"] / etf_detail["Parts totales"].replace(0, float("nan"))
        
        # Mask if requested
        display_etf = etf_detail.copy()
        if hide_amounts:
            display_etf["Investi (€)"] = "••••"
            display_etf["Parts totales"] = "••••"
            display_etf["Prix moy./part (€)"] = "••••"
            display_etf["Valeur/part actuelle (€)"] = "••••"
            st.dataframe(display_etf, use_container_width=True, hide_index=True)
        else:
            st.dataframe(
                display_etf.style.format({
                    "Investi (€)": "{:,.2f}", "Parts totales": "{:.4f}",
                    "Prix moy./part (€)": "{:.2f}", "Valeur/part actuelle (€)": "{:.2f}"
                }).background_gradient(subset=["Investi (€)"], cmap="Greens"),
                use_container_width=True, hide_index=True
            )
            
        st.markdown("---")
        st.subheader("Saveback & Interets")
        c1, c2 = st.columns(2)
        with c1:
            sav_monthly = savings.groupby("month")["amount"].sum().reset_index()
            fig_sav = go.Figure(go.Bar(x=sav_monthly["month"], y=sav_monthly["amount"],
                marker_color="#a855f7",
                hovertemplate="<b>%{x}</b><br>" + ("••••" if hide_amounts else "%{y:.2f} €") + "<extra></extra>"))
            fig_sav.update_layout(title="Saveback mensuel (€)", template="plotly_dark",
                plot_bgcolor="rgba(15,23,42,0.5)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), height=280,
                xaxis=dict(tickangle=45), yaxis=dict(showticklabels=not hide_amounts), margin=dict(l=0,r=0,t=40,b=60))
            st.plotly_chart(fig_sav, use_container_width=True)
        with c2:
            int_monthly = interests.groupby("month")["amount"].sum().reset_index()
            fig_int = go.Figure(go.Bar(x=int_monthly["month"], y=int_monthly["amount"],
                marker_color="#fbbf24",
                hovertemplate="<b>%{x}</b><br>" + ("••••" if hide_amounts else "%{y:.2f} €") + "<extra></extra>"))
            fig_int.update_layout(title="Intérêts mensuels (€)", template="plotly_dark",
                plot_bgcolor="rgba(15,23,42,0.5)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"), height=280,
                xaxis=dict(tickangle=45), yaxis=dict(showticklabels=not hide_amounts), margin=dict(l=0,r=0,t=40,b=60))
            st.plotly_chart(fig_int, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — DÉPENSES
# ════════════════════════════════════════════════════════════════
with main_tab2:
    tab_dep1, tab_dep2, tab_dep3 = st.tabs(["Sankey", "Par mois", "Top marchands"])

    with tab_dep1:
        st.markdown("##### Flux de trésorerie — Entrées vers dépenses")
        
        # Detail controller
        sankey_detail = st.radio(
            "Niveau de détail du diagramme de Sankey :",
            ["Categories uniquement", "Categories & Top Marchands"],
            horizontal=True
        )

        cat_totals = cards.groupby("category_label")["amount"].apply(lambda x: x.abs().sum())
        cat_totals = cat_totals[cat_totals > 5].sort_values(ascending=False)

        # Build Sankey nodes & links
        source_node = "Depenses carte"
        cats = cat_totals.index.tolist()
        
        if "Top Marchands" in sankey_detail:
            merchant_data = cards[cards["category_label"].isin(cat_totals.index)].copy()
            merchant_data["merchant_clean"] = merchant_data["name"].str.strip().str.split(r"\s{2,}").str[0].str[:20]
            top_merchants = (merchant_data.groupby(["category_label", "merchant_clean"])["amount"]
                             .apply(lambda x: x.abs().sum())
                             .reset_index())
            top_merchants.columns = ["cat", "merchant", "amount"]
            top_merchants = top_merchants[top_merchants["amount"] > 5]
            top_merchants_per_cat = (top_merchants.sort_values("amount", ascending=False)
                                     .groupby("cat").head(3))
            
            merch_list = (top_merchants_per_cat["merchant"] + " (" + top_merchants_per_cat["cat"] + ")").unique().tolist()
            all_nodes = [source_node] + cats + merch_list
        else:
            all_nodes = [source_node] + cats

        node_idx = {n: i for i, n in enumerate(all_nodes)}
        sources, targets, values, link_colors = [], [], [], []
        cat_palette = px.colors.qualitative.Set3

        # First level: Source -> Category
        for i, (cat, total) in enumerate(cat_totals.items()):
            color = cat_palette[i % len(cat_palette)]
            sources.append(node_idx[source_node])
            targets.append(node_idx[cat])
            values.append(total)
            link_colors.append(color.replace(")", ",0.4)").replace("rgb", "rgba"))

        # Second level: Category -> Merchant (only if requested)
        if "Top Marchands" in sankey_detail:
            for _, row in top_merchants_per_cat.iterrows():
                merch_key = f"{row['merchant']} ({row['cat']})"
                if merch_key in node_idx and row["cat"] in node_idx:
                    sources.append(node_idx[row["cat"]])
                    targets.append(node_idx[merch_key])
                    values.append(row["amount"])
                    cat_i = cats.index(row["cat"]) if row["cat"] in cats else 0
                    c = cat_palette[cat_i % len(cat_palette)]
                    link_colors.append(c.replace(")", ",0.3)").replace("rgb", "rgba"))

        # Node colors
        if "Top Marchands" in sankey_detail:
            node_colors = ["#10b981"] + [cat_palette[i % len(cat_palette)] for i in range(len(cats))] + ["#475569"] * len(merch_list)
        else:
            node_colors = ["#10b981"] + [cat_palette[i % len(cat_palette)] for i in range(len(cats))]

        fig_sankey = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=20, thickness=20,
                label=all_nodes,
                color=node_colors,
                hovertemplate="%{label}<br>" + ("<b>••••</b>" if hide_amounts else "<b>%{value:,.2f} €</b>") + "<extra></extra>"
            ),
            link=dict(
                source=sources, target=targets, value=values,
                color=link_colors,
                hovertemplate="%{source.label} → %{target.label}<br>" + ("<b>••••</b>" if hide_amounts else "<b>%{value:,.2f} €</b>") + "<extra></extra>"
            )
        ))
        fig_sankey.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(15,23,42,0.8)",
            font=dict(color="#e2e8f0", size=13), height=600,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_sankey, use_container_width=True)

    with tab_dep2:
        monthly_cats = (cards.groupby(["month", "category_label"])["amount"]
                        .apply(lambda x: x.abs().sum()).reset_index())
        monthly_cats.columns = ["month", "categorie", "montant"]

        fig_stack = px.bar(monthly_cats, x="month", y="montant", color="categorie",
                           color_discrete_sequence=px.colors.qualitative.Set3,
                           labels={"montant": "€", "month": "", "categorie": "Categorie"})
        fig_stack.update_layout(
            title="Dépenses par catégorie et par mois",
            template="plotly_dark", plot_bgcolor="rgba(15,23,42,0.5)",
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
            barmode="stack", height=450, xaxis=dict(tickangle=45),
            yaxis=dict(showticklabels=not hide_amounts),
            legend=dict(orientation="h", y=-0.3),
            margin=dict(l=0, r=0, t=50, b=100)
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        # Evolution dépenses totales mensuelles
        monthly_total = cards.groupby("month")["amount"].apply(lambda x: x.abs().sum()).reset_index()
        monthly_total.columns = ["month", "total"]
        monthly_total["moyenne_mobile"] = monthly_total["total"].rolling(3, min_periods=1).mean()
        fig_line = go.Figure()
        fig_line.add_trace(go.Bar(x=monthly_total["month"], y=monthly_total["total"],
            name="Dépenses", marker_color="#ef4444", opacity=0.7))
        fig_line.add_trace(go.Scatter(x=monthly_total["month"], y=monthly_total["moyenne_mobile"],
            name="Moy. 3 mois", line=dict(color="#fbbf24", width=2.5)))
        fig_line.update_layout(
            title="Total dépenses mensuelles + moyenne mobile",
            template="plotly_dark", plot_bgcolor="rgba(15,23,42,0.5)",
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
            height=320, barmode="overlay", xaxis=dict(tickangle=45),
            yaxis=dict(showticklabels=not hide_amounts),
            margin=dict(l=0, r=0, t=50, b=60)
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with tab_dep3:
        cards_clean = cards.copy()
        cards_clean["merchant_clean"] = cards_clean["name"].str.strip().str.split(r"\s{2,}").str[0].str[:25]
        top20 = (cards_clean.groupby(["merchant_clean", "category_label"])
                 .agg(Total=("amount", lambda x: x.abs().sum()), Nb=("amount", "count"))
                 .reset_index().sort_values("Total", ascending=False).head(20))
        top20.columns = ["Marchand", "Categorie", "Total (€)", "Nb transactions"]

        colors_top = ["#ef4444" if i < 3 else "#f97316" if i < 7 else "#64748b"
                      for i in range(len(top20))]
        fig_top = go.Figure(go.Bar(
            x=top20["Total (€)"][::-1],
            y=(top20["Marchand"] + " " + top20["Categorie"])[::-1],
            orientation="h", marker_color=colors_top[::-1],
            hovertemplate="<b>%{y}</b><br>" + ("•••• €" if hide_amounts else "%{x:,.2f} €") + "<extra></extra>"
        ))
        fig_top.update_layout(
            title="Top 20 marchands (total cumulé)",
            template="plotly_dark", plot_bgcolor="rgba(15,23,42,0.5)",
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
            height=580, xaxis=dict(title="€ total", showgrid=True, gridcolor="rgba(102,126,234,0.1)", showticklabels=not hide_amounts),
            yaxis=dict(showgrid=False), margin=dict(l=10, r=10, t=50, b=0)
        )
        st.plotly_chart(fig_top, use_container_width=True)

        st.markdown("---")
        st.subheader("Toutes les dépenses")
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            cats_filter = st.multiselect("Filtrer par catégorie", sorted(cards["category_label"].unique()),
                                         default=sorted(cards["category_label"].unique()))
        with col_filter2:
            months_filter = st.multiselect("Filtrer par mois", sorted(cards["month"].unique()),
                                           default=sorted(cards["month"].unique())[-3:])
        filtered = cards[cards["category_label"].isin(cats_filter) & cards["month"].isin(months_filter)]
        show = filtered[["date", "name", "category_label", "amount"]].copy()
        show.columns = ["Date", "Marchand", "Categorie", "Montant (€)"]
        show["Montant (€)"] = show["Montant (€)"].abs()
        
        display_show = show.copy()
        if hide_amounts:
            display_show["Montant (€)"] = "••••"
            st.dataframe(display_show.sort_values("Date", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.dataframe(display_show.sort_values("Date", ascending=False).style.format({"Montant (€)": "{:.2f}"}),
                         use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — SIMULATEUR IFU (FISC)
# ════════════════════════════════════════════════════════════════
with main_tab3:
    st.markdown("### Simulateur d'Aide à la Déclaration d'Impôts (IFU)")
    st.markdown("""
    Puisque Trade Republic est un établissement basé en Allemagne (étranger), vous devez déclarer vos intérêts 
    et votre compte espèces. Ce simulateur extrait les données correspondantes de votre fichier de transactions 
    pour vous aider à remplir votre déclaration de revenus française (Formulaire 2042 et 3916).
    """)
    
    st.info("Note Légale : Ce simulateur est fourni à titre indicatif pour vous aider dans vos démarches fiscales. Les montants calculés correspondent aux transactions détectées dans votre fichier de transactions exporté.")

    # Group interests by calendar year
    interests_copy = interests.copy()
    interests_copy["year"] = interests_copy["date"].dt.year
    years = sorted(interests_copy["year"].unique(), reverse=True)
    
    if not years:
        st.warning("Aucune transaction d'intérêts rémunérés (INTEREST_PAYMENT) n'a été détectée dans l'export.")
    else:
        for year in years:
            st.markdown(f"#### Revenus perçus en {year} (Déclaration en {year+1})")
            
            y_interests = interests_copy[interests_copy["year"] == year]
            net_int = y_interests["amount"].sum()
            tax_paid = y_interests["tax"].abs().sum()
            gross_int = net_int + tax_paid
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="tax-card">
                    <h5>Formulaire 2042 — Déclaration Principale</h5>
                    <p>Déclarez ces montants dans la section <i>Revenus des valeurs et capitaux mobiliers</i> :</p>
                    <div class="tax-box">
                        <div class="tax-label">Case <b>2TR</b> (Intérêts bruts soumis au barème ou flat tax)</div>
                        <div class="tax-value">{fmt(gross_int)}</div>
                    </div>
                    <div class="tax-box">
                        <div class="tax-label">Case <b>2CK</b> (Prélèvement forfaitaire non libératoire déjà versé)</div>
                        <div class="tax-value">{fmt(tax_paid)}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="tax-card" style="border-color: #3b82f6;">
                    <h5 style="color: #3b82f6;">Synthèse des Gains Rémunérés</h5>
                    <table style="width:100%; border-collapse: collapse; margin-top: 10px; font-size:13px;">
                        <tr>
                            <td style="padding: 5px 0; color: #9ca3af;">Intérêts Nets encaissés :</td>
                            <td style="padding: 5px 0; text-align: right; font-weight: bold;">{fmt(net_int)}</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px 0; color: #9ca3af;">Prélèvement à la source (Acompte) :</td>
                            <td style="padding: 5px 0; text-align: right; color: #f87171;">- {fmt(tax_paid)}</td>
                        </tr>
                        <tr style="border-top: 1px solid #374151;">
                            <td style="padding: 8px 0; font-weight: bold;">Intérêts Brut de Taxe :</td>
                            <td style="padding: 8px 0; text-align: right; font-weight: bold; color: #10b981;">{fmt(gross_int)}</td>
                        </tr>
                    </table>
                    <p style="font-size: 11px; color: #9ca3af; margin-top: 15px;">
                        <i>Le prélèvement à la source effectué par Trade Republic fait office d'acompte (flat-tax acomptes). La case 2CK génère un crédit d'impôt équivalent pour éviter la double imposition.</i>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
    st.markdown("---")
    st.markdown("### Formulaire 3916 — Déclaration de Compte à l'Étranger")
    st.markdown("""
    Tout compte ouvert à l'étranger doit obligatoirement être déclaré chaque année en même temps que vos revenus via le formulaire 3916. 
    Voici les informations pré-remplies relatives à votre compte Trade Republic :
    """)
    
    with st.expander("Afficher les détails à recopier pour le formulaire 3916"):
        st.markdown(f"""
        | Champ sur le formulaire 3916 | Valeur à saisir |
        | :--- | :--- |
        | **Intitulé du compte** | Compte d'espèces Trade Republic |
        | **Désignation de l'organisme d'accueil** | Trade Republic Bank GmbH |
        | **Pays d'accueil** | Allemagne (Germany) |
        | **Adresse de l'établissement** | Brunnenstraße 19-21, 10119 Berlin |
        | **Caractéristiques du compte** | Compte courant / d'épargne rémunéré non-courant |
        | **Numéro de compte / IBAN** | *Vérifiez sur votre application TR (commence par DE)* |
        """)
        
        st.info("Attention : L'amende en cas de non-déclaration d'un compte à l'étranger est de 1 500 € par an et par compte. N'oubliez pas de cocher la case **8UU** sur votre déclaration principale 2042.")
