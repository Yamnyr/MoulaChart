import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Calculatrice d'intérêts composés", page_icon="💰", layout="wide")

st.title("Calculatrice d'intérêts composés avancée")
st.markdown("Simulez la croissance de votre patrimoine avec des scénarios réalistes")
st.markdown("---")

# Sidebar pour les paramètres avancés
with st.sidebar:
    st.title("💰 Calculatrice d'intérêts composés")
    st.markdown("---")
    st.header("Paramètres avancés")

    mode_fiscalite = st.checkbox("Inclure la fiscalité", value=False)
    if mode_fiscalite:
        taux_imposition = st.slider("Taux d'imposition (%)", 0.0, 50.0, 30.0, 0.5)

    mode_inflation = st.checkbox("Ajuster à l'inflation", value=False)
    if mode_inflation:
        taux_inflation = st.slider("Taux d'inflation annuel (%)", 0.0, 10.0, 2.0, 0.1)

    progression_epargne = st.checkbox("Progression de l'épargne", value=False)
    if progression_epargne:
        taux_progression = st.slider("Augmentation annuelle de l'épargne (%)", 0.0, 10.0, 2.0, 0.5)

    mode_comparaison = st.checkbox("Comparer plusieurs scénarios", value=False)

# Formulaire de calcul
with st.form("calcul_interets"):
    col1, col2 = st.columns(2)

    with col1:
        capital_initial = st.number_input(
            "Capital initial (€)",
            min_value=0.0,
            value=10000.0,
            step=1000.0,
            help="Montant de départ de votre investissement"
        )

        epargne_mensuelle = st.number_input(
            "Épargne mensuelle (€)",
            min_value=0.0,
            value=200.0,
            step=50.0,
            help="Montant que vous comptez épargner chaque mois"
        )

    with col2:
        horizon = st.number_input(
            "Horizon de placement (années)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Durée de votre investissement en années"
        )

        taux_interet = st.number_input(
            "Taux d'intérêt annuel (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            help="Rendement annuel espéré de votre investissement"
        )

    if mode_comparaison:
        st.markdown("**Scénarios de comparaison**")
        col3, col4 = st.columns(2)
        with col3:
            taux_optimiste = st.number_input("Taux optimiste (%)", 0.0, 20.0, taux_interet + 2.0, 0.1)
        with col4:
            taux_pessimiste = st.number_input("Taux pessimiste (%)", 0.0, 20.0, max(0.0, taux_interet - 2.0), 0.1)

    submitted = st.form_submit_button("Calculer", width='stretch')


def calculer_evolution(capital_init, epargne_mens, taux_annuel, horizon_ans, taux_prog=0, taux_infl=0, taux_impot=0):
    """Calcule l'évolution du capital avec tous les paramètres"""
    taux_mensuel = taux_annuel / 100 / 12
    nb_mois = horizon_ans * 12

    mois = []
    capital_verse = []
    interets_bruts = []
    interets_nets = []
    total_nominal = []
    total_reel = []

    capital_actuel = capital_init
    total_verse = capital_init
    epargne_actuelle = epargne_mens

    for m in range(nb_mois + 1):
        annee = m / 12
        mois.append(annee)
        capital_verse.append(total_verse)

        # Calcul des intérêts
        interets_cumules_brut = capital_actuel - total_verse
        impots_cumules = interets_cumules_brut * (taux_impot / 100) if taux_impot > 0 else 0
        interets_cumules_net = interets_cumules_brut - impots_cumules

        interets_bruts.append(interets_cumules_brut)
        interets_nets.append(interets_cumules_net)
        total_nominal.append(capital_actuel)

        # Ajustement à l'inflation
        if taux_infl > 0:
            facteur_inflation = (1 + taux_infl / 100) ** annee
            total_reel.append(capital_actuel / facteur_inflation)
        else:
            total_reel.append(capital_actuel)

        if m < nb_mois:
            # Calcul des intérêts mensuels (après impôts si applicable)
            interets_mois = capital_actuel * taux_mensuel
            if taux_impot > 0:
                interets_mois *= (1 - taux_impot / 100)

            capital_actuel = capital_actuel + interets_mois + epargne_actuelle
            total_verse += epargne_actuelle

            # Progression annuelle de l'épargne
            if taux_prog > 0 and (m + 1) % 12 == 0:
                epargne_actuelle *= (1 + taux_prog / 100)

    return {
        'mois': mois,
        'capital_verse': capital_verse,
        'interets_bruts': interets_bruts,
        'interets_nets': interets_nets,
        'total_nominal': total_nominal,
        'total_reel': total_reel
    }


if submitted:
    # Calculs du scénario principal
    taux_prog_val = taux_progression if progression_epargne else 0
    taux_infl_val = taux_inflation if mode_inflation else 0
    taux_impot_val = taux_imposition if mode_fiscalite else 0

    resultats = calculer_evolution(
        capital_initial,
        epargne_mensuelle,
        taux_interet,
        horizon,
        taux_prog_val,
        taux_infl_val,
        taux_impot_val
    )

    # Résultats finaux
    capital_final = resultats['total_nominal'][-1]
    capital_final_reel = resultats['total_reel'][-1]
    total_verse_final = resultats['capital_verse'][-1]
    interets_final = resultats['interets_nets'][-1] if mode_fiscalite else resultats['interets_bruts'][-1]

    # Affichage des résultats
    st.markdown("---")
    st.subheader("Résultats de la simulation")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Capital final",
            value=f"{capital_final:,.0f} €",
            delta=f"+{interets_final:,.0f} €"
        )

    with col2:
        st.metric(
            label="Total versé",
            value=f"{total_verse_final:,.0f} €"
        )

    with col3:
        rendement_pct = (interets_final / total_verse_final) * 100 if total_verse_final > 0 else 0
        st.metric(
            label="Intérêts générés",
            value=f"{interets_final:,.0f} €",
            delta=f"{rendement_pct:.1f}% de gain"
        )

    with col4:
        if mode_inflation:
            perte_pouvoir_achat = capital_final - capital_final_reel
            st.metric(
                label="Capital réel (après inflation)",
                value=f"{capital_final_reel:,.0f} €",
                delta=f"-{perte_pouvoir_achat:,.0f} €",
                delta_color="inverse"
            )
        else:
            rentabilite_annuelle = ((capital_final / total_verse_final) ** (
                        1 / horizon) - 1) * 100 if total_verse_final > 0 else 0
            st.metric(
                label="Rentabilité annualisée",
                value=f"{rentabilite_annuelle:.2f}%"
            )

    # Graphiques
    st.markdown("---")

    if mode_comparaison:
        st.subheader("Comparaison des scénarios")

        # Calculs des scénarios alternatifs
        res_optimiste = calculer_evolution(capital_initial, epargne_mensuelle, taux_optimiste, horizon, taux_prog_val,
                                           taux_infl_val, taux_impot_val)
        res_pessimiste = calculer_evolution(capital_initial, epargne_mensuelle, taux_pessimiste, horizon, taux_prog_val,
                                            taux_infl_val, taux_impot_val)

        fig = go.Figure()

        # Scénario pessimiste
        fig.add_trace(go.Scatter(
            x=res_pessimiste['mois'],
            y=res_pessimiste['total_reel'] if mode_inflation else res_pessimiste['total_nominal'],
            name=f"Scénario pessimiste ({taux_pessimiste}%)",
            line=dict(color='#ef4444', width=2, dash='dot'),
            mode='lines'
        ))

        # Scénario principal
        fig.add_trace(go.Scatter(
            x=resultats['mois'],
            y=resultats['total_reel'] if mode_inflation else resultats['total_nominal'],
            name=f"Scénario attendu ({taux_interet}%)",
            line=dict(color='#8b5cf6', width=3),
            mode='lines'
        ))

        # Scénario optimiste
        fig.add_trace(go.Scatter(
            x=res_optimiste['mois'],
            y=res_optimiste['total_reel'] if mode_inflation else res_optimiste['total_nominal'],
            name=f"Scénario optimiste ({taux_optimiste}%)",
            line=dict(color='#10b981', width=2, dash='dot'),
            mode='lines'
        ))

        # Capital versé
        fig.add_trace(go.Scatter(
            x=resultats['mois'],
            y=resultats['capital_verse'],
            name="Capital versé",
            line=dict(color='#94a3b8', width=2, dash='dash'),
            mode='lines'
        ))

        fig.update_layout(
            xaxis_title="Années",
            yaxis_title="Montant (€)" + (" - Pouvoir d'achat actuel" if mode_inflation else ""),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            template="plotly_white"
        )

        st.plotly_chart(fig, config={"responsive": True})

        # Tableau comparatif
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Scénario pessimiste", f"{res_pessimiste['total_nominal'][-1]:,.0f} €")
        with col2:
            st.metric("Scénario attendu", f"{capital_final:,.0f} €")
        with col3:
            st.metric("Scénario optimiste", f"{res_optimiste['total_nominal'][-1]:,.0f} €")

    else:
        st.subheader("Évolution du patrimoine")

        fig = go.Figure()

        # Courbe de l'argent versé
        fig.add_trace(go.Scatter(
            x=resultats['mois'],
            y=resultats['capital_verse'],
            name="Capital versé",
            line=dict(color='#3b82f6', width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))

        # Courbe des intérêts
        interets_affichage = resultats['interets_nets'] if mode_fiscalite else resultats['interets_bruts']
        fig.add_trace(go.Scatter(
            x=resultats['mois'],
            y=interets_affichage,
            name="Intérêts générés" + (" (nets)" if mode_fiscalite else ""),
            line=dict(color='#10b981', width=3),
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.2)'
        ))

        # Courbe du total
        total_affichage = resultats['total_reel'] if mode_inflation else resultats['total_nominal']
        fig.add_trace(go.Scatter(
            x=resultats['mois'],
            y=total_affichage,
            name="Patrimoine total" + (" (réel)" if mode_inflation else ""),
            line=dict(color='#8b5cf6', width=3, dash='dash'),
            mode='lines'
        ))

        fig.update_layout(
            xaxis_title="Années",
            yaxis_title="Montant (€)" + (" - Pouvoir d'achat actuel" if mode_inflation else ""),
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            template="plotly_white"
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')

        st.plotly_chart(fig, config={"responsive": True})

    # Graphique en camembert
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Composition du patrimoine final")

        fig_pie = go.Figure(data=[go.Pie(
            labels=['Capital versé', 'Intérêts générés'],
            values=[total_verse_final, interets_final],
            marker=dict(colors=['#3b82f6', '#10b981']),
            hole=0.4,
            textinfo='label+percent',
            textposition='outside'
        )])

        fig_pie.update_layout(
            showlegend=False,
            height=300,
            margin=dict(t=0, b=0, l=0, r=0)
        )

        st.plotly_chart(fig_pie, config={"responsive": True})

    with col2:
        st.subheader("Objectifs de patrimoine")

        objectifs = [50000, 100000, 250000, 500000, 1000000]
        objectif_atteint = None

        for obj in objectifs:
            if capital_final >= obj:
                objectif_atteint = obj

        if objectif_atteint:
            st.success(f"✅ Objectif de {objectif_atteint:,} € atteint!")
            prochain_objectif = next((o for o in objectifs if o > capital_final), None)
            if prochain_objectif:
                manque = prochain_objectif - capital_final
                st.info(f"🎯 Prochain objectif: {prochain_objectif:,} € (encore {manque:,.0f} €)")
        else:
            premier_objectif = objectifs[0]
            manque = premier_objectif - capital_final
            st.info(f"🎯 Premier objectif: {premier_objectif:,} € (encore {manque:,.0f} €)")

        # Calcul du temps pour doubler le capital
        if taux_interet > 0:
            annees_doublement = 72 / taux_interet  # Règle de 72
            st.metric(
                "Temps pour doubler votre capital",
                f"{annees_doublement:.1f} ans",
                help="Calculé selon la règle de 72"
            )

    # Tableau détaillé par année
    st.markdown("---")
    st.subheader("Détail année par année")

    annees_data = []
    for annee in range(horizon + 1):
        idx = annee * 12
        if idx < len(resultats['mois']):
            row = {
                "Année": annee,
                "Capital versé (€)": f"{resultats['capital_verse'][idx]:,.0f}",
                "Intérêts (€)": f"{(resultats['interets_nets'][idx] if mode_fiscalite else resultats['interets_bruts'][idx]):,.0f}",
                "Total nominal (€)": f"{resultats['total_nominal'][idx]:,.0f}"
            }
            if mode_inflation:
                row["Total réel (€)"] = f"{resultats['total_reel'][idx]:,.0f}"
            annees_data.append(row)

    df_annees = pd.DataFrame(annees_data)
    st.dataframe(df_annees, width='stretch', hide_index=True)

else:
    st.info("Remplissez le formulaire ci-dessus et cliquez sur 'Calculer' pour voir vos résultats.")
