# 📈 MoulaChart

**MoulaChart** est une application Streamlit interactive permettant de comparer les performances boursières de plusieurs entreprises en temps réel via l’API **Yahoo Finance**.


---

## Fonctionnalités

- Sélection dynamique des tickers du **S&P 500**
- Visualisation interactive avec **Plotly**
- Option de **normalisation** (base 100)
- Calcul automatique de :
  - Performance (%)
  - Volatilité (%)
  - Rendement moyen (%)
- Interface sombre personnalisée (noir + vert billet)
- Mise en cache automatique des tickers pour rapidité

---

## 🛠️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Yamnyr/MoulaChart.git
cd MoulaChart
```

### 2. Créer un environnement virtuel

#### Avec **conda** :
```bash
conda create -n finance_app python=3.11
conda activate finance_app
```

#### Ou avec **venv** :
```bash
python -m venv .venv
source .venv/bin/activate    # (ou .venv\Scripts\activate sous Windows)
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## ▶️ Lancer l’application

```bash
streamlit run app.py
```

Ensuite, ouvre ton navigateur sur :  
👉 **http://localhost:8501**

---

## 🧾 Exemple d’utilisation

1. Sélectionne plusieurs tickers (ex. `AAPL`, `MSFT`, `NVDA`)  
2. Choisis la période (`6 mois`, `1 an`, etc.)  
3. Visualise instantanément la **performance normalisée** sur un graphique interactif  
4. Consulte le tableau récapitulatif des statistiques financières

---

## 🧩 Structure du projet

```
MoulaChart/
│
├── app.py                  # Application principale Streamlit
├── requirements.txt        # Liste des dépendances
├── .streamlit/
│   └── config.toml         # Thème vert & noir personnalisé
└── README.md               # Documentation du projet
```

---
