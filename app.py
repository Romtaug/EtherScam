import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import squarify
import locale
import joblib
import os
import unicodedata
import difflib
from PIL import Image

# Chargement du modèle et configuration locale
model = joblib.load("models/xgb_best_model.pkl")

if "Etherscam" not in st.session_state:
    st.session_state["Etherscam"] = "🚨 Analyse Wallet"  # ou autre valeur par défaut
###################################################################################################################################################################################################################

def analyse_wallet_complete(features, df_eth, lifetime_days, address):
    diagnostics = []
    profile_tags = []
    danger_score = 0

    # Variables clés
    sent_count = features["Sent tnx"]
    recv_count = features["Received Tnx"]
    sent_sum = features["total Ether sent"]
    recv_sum = features["total ether received"]
    balance = features["total ether balance"]
    sent_to = features["Unique Sent To Addresses"]
    recv_from = features["Unique Received From Addresses"]
    redistribution_ratio = 1 - (balance / (recv_sum + 1e-6))
    avg_val_sent = features["avg val sent"]
    avg_val_received = features["avg val received"]
    tx_ratio = sent_count / (recv_count + 1e-6)

    # ⏱️ Analyse temporelle
    if lifetime_days < 15:
        diagnostics.append(f"🚨 Wallet actif depuis **{lifetime_days} jours seulement**. Très jeune, typique des scams temporaires.")
        danger_score += 3
    elif lifetime_days < 90:
        diagnostics.append(f"📅 Wallet jeune (≈ {lifetime_days} jours). Historique limité, attention au contexte.")
        danger_score += 2
    elif lifetime_days < 365:
        diagnostics.append(f"🗓️ Wallet actif depuis moins d’un an : **{lifetime_days} jours**. Antériorité moyenne.")
        danger_score += 1
    else:
        diagnostics.append(f"🕰️ Wallet actif depuis **{lifetime_days} jours**. Ancienneté rassurante.")
        if sent_sum > 300 and recv_sum > 300:
            diagnostics.append("📊 Forte activité financière sur longue période. Le wallet gère des volumes importants sur le long terme.")
        elif sent_sum > 300 and balance < 1:
            diagnostics.append("🚨 Wallet très actif et ancien, mais totalement vidé. Dump / blanchiment / fuite de capitaux ?")
            danger_score += 2
        elif recv_sum > 300 and sent_sum < 1:
            diagnostics.append("🧲 Wallet ancien qui a reçu beaucoup sans rien envoyer. Potentiel piège à fonds ou cold storage.")
            danger_score += 3

    # ⏰ Comportement horaire
    df_eth["hour"] = df_eth["timeStamp"].dt.hour
    df_eth["weekday"] = df_eth["timeStamp"].dt.dayofweek
    most_active_hour = df_eth["hour"].mode()[0]
    most_active_day = df_eth["weekday"].mode()[0]
    hour_freq = df_eth["hour"].value_counts(normalize=True).max()

    if hour_freq > 0.5:
        diagnostics.append(f"🕔 Plus de 50% des transactions à **{most_active_hour}h**. Probable script ou bot.")
        danger_score += 1
    else:
        diagnostics.append(f"📈 Activité répartie : heure dominante = **{most_active_hour}h**.")

    diagnostics.append(f"📅 Jour dominant = **{['Lun','Mar','Mer','Jeu','Ven','Sam','Dim'][most_active_day]}**.")

    # 📤📥 Ratio tx
    if recv_count == 0 and sent_count > 0:
        diagnostics.append("📤 Le wallet **n’a jamais reçu**, mais envoie. Relais ? Burner ?")
        danger_score += 4
    elif sent_count == 0 and recv_count > 0:
        diagnostics.append("📥 Le wallet **ne fait que recevoir**. Attire des fonds ?")
        danger_score += 4
    else:
        if tx_ratio > 10:
            diagnostics.append("📈 Ratio tx >10 : ce wallet envoie beaucoup plus qu’il ne reçoit.")
            danger_score += 2
        elif tx_ratio < 0.1:
            diagnostics.append("📉 Ratio tx <0.1 : reçoit beaucoup, envoie très peu. Collector ?")
            danger_score += 3
        elif tx_ratio > 3:
            diagnostics.append(f"📤 {tx_ratio:.2f}x plus d’envois que de réceptions.")
            danger_score += 1
        else:
            diagnostics.append("⚖️ Ratio tx équilibré.")

    # 💸 Solde final
    if balance < 0.0001 and recv_sum > 1:
        diagnostics.append("💸 Wallet vidé malgré de grosses réceptions. Suspect.")
        danger_score += 3
    elif balance < 0.01:
        diagnostics.append("🔄 Solde très faible. Wallet temporaire ?")
        danger_score += 1
    else:
        if redistribution_ratio > 0.95:
            diagnostics.append(f"🔁 Redistribution : **{redistribution_ratio*100:.2f}%** des fonds reçus sont sortis.")
            danger_score += 1
        elif redistribution_ratio < 0.1:
            diagnostics.append("🔒 Wallet conserve ses fonds. HODL ?")

    # 📤 Fragmentation
    if avg_val_sent < avg_val_received * 0.5:
        diagnostics.append(f"📤 Envois fragmentés ({avg_val_sent:.2f} ETH vs {avg_val_received:.2f} ETH reçus).")
        danger_score += 1

    # 💥 Grosse tx
    if features["max value received"] > 250:
        diagnostics.append("🧨 Transaction >250 ETH détectée. Gros flux ?")
        danger_score += 2

    # 🔁 Pattern
    if sent_count > 50 and recv_count < 10:
        diagnostics.append("📡 Pattern flooder : beaucoup d’envois, peu de réceptions.")
        profile_tags.append("Flooder")
        danger_score += 2

    if sent_to > 50 and balance < 1 and lifetime_days < 90:
        diagnostics.append("🔥 Burner : jeune, balance vide, envoie à >50 adresses.")
        profile_tags.append("Burner")
        danger_score += 3

    if recv_from > 30 and sent_count == 0:
        diagnostics.append("🧲 Collector : reçoit beaucoup, n'envoie rien.")
        profile_tags.append("Collector")
        danger_score += 2

    if recv_sum > 100 and sent_sum < 1:
        diagnostics.append("🚩 Reçoit >100 ETH sans jamais renvoyer. Comportement SCAM-like.")
        profile_tags.append("Scam-like")
        danger_score += 4

    # 📊 Écart-type
    val_std = df_eth["eth_value"].std()
    if val_std > 10:
        diagnostics.append(f"📉 Volatilité élevée : écart-type = {val_std:.2f} ETH.")
        danger_score += 1

    # 😴 Faible activité
    if (recv_count + sent_count) < 5:
        diagnostics.append("💤 Moins de 5 tx. Dormant ? Test ?")
        profile_tags.append("Dormant")

    # ⚠️ Tx nulles
    if features["min val sent"] == 0.0 or features["min value received"] == 0.0:
        diagnostics.append("⚠️ Transactions nulles (0 ETH) détectées.")
        danger_score += 1

    return danger_score, diagnostics, profile_tags



###############################################################################################################################

st.set_page_config(page_title="EtherScam", page_icon="🚨")

# 📌 Sidebar navigation
page = st.sidebar.radio(
    "EtherScam",
    ["🚨 Analyse Wallet", "🤖 Prédiction IA", "⚙️ Générateur de Données", "📝 À propos"],
    index=["🚨 Analyse Wallet", "🤖 Prédiction IA", "⚙️ Générateur de Données", "📝 À propos"].index(st.session_state["Etherscam"]),
    key="EtherScam"
)
#############################################################################################################################################################
# ⚙️ Générateur interactif de dataset à partir d'adresses Ethereum
if page == "⚙️ Générateur de Données":
    st.title("⚙️ Générateur de Données")
    st.markdown("""
    Ce générateur interactif permet de **créer un fichier CSV à partir de plusieurs adresses Ethereum**.

    - Chaque adresse est analysée via l’API d’Etherscan
    - Les colonnes sont automatiquement extraites pour correspondre aux besoins du modèle IA
    - Tu peux visualiser, supprimer les doublons, puis télécharger le fichier prêt à l’emploi

    C’est l’étape idéale pour **constituer un jeu de données d’entraînement ou de test personnalisé**.
    """)

    api_key = "GAK4SSJCDJDURKJMB8RM62QDW84HJZT57T"

    colonnes_features = [
        "Address",
        "Avg min between sent tnx",
        "Avg min between received tnx",
        "Time Diff between first and last (Mins)",
        "Sent tnx",
        "Received Tnx",
        "Unique Received From Addresses",
        "Unique Sent To Addresses",
        "min value received",
        "max value received",
        "avg val received",
        "min val sent",
        "max val sent",
        "avg val sent",
        "total transactions (including tnx to create contract",
        "total Ether sent",
        "total ether received",
        "total ether balance"
    ]

    if "wallet_dataset" not in st.session_state:
        st.session_state.wallet_dataset = pd.DataFrame(columns=colonnes_features)

    address_input = st.text_input(
        "➕ Adresse Ethereum à ajouter au dataset",
        value="0xD0cC2B24980CBCCA47EF755Da88B220a82291407"
    )

    col1, col2 = st.columns(2)

    with col1:
        add_clicked = st.button("🔍 Ajouter au dataset", use_container_width=True)

    with col2:
        reset_clicked = st.button("🔄 Réinitialiser le dataset", use_container_width=True)

    if reset_clicked:
        st.session_state.wallet_dataset = pd.DataFrame(columns=colonnes_features)

    if add_clicked and address_input:
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address_input}&startblock=0&endblock=99999999&sort=asc&apikey={api_key}"
        response = requests.get(url)
        data = response.json()

        if data["status"] != "1":
            st.error("❌ Adresse invalide ou aucune transaction trouvée.")
        else:
            df = pd.DataFrame(data["result"])
            df["datetime"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s")
            df["timeStamp"] = df["datetime"]
            df["eth_value"] = df["value"].astype(float) / 1e18

            sent = df[df["from"].str.lower() == address_input.lower()]
            received = df[df["to"].str.lower() == address_input.lower()]
            df_eth = pd.concat([sent, received]).sort_values("timeStamp")

            features = {
                "Address": address_input,
                "Avg min between sent tnx": sent["timeStamp"].diff().dt.total_seconds().div(60).mean() if len(sent) > 1 else 0,
                "Avg min between received tnx": received["timeStamp"].diff().dt.total_seconds().div(60).mean() if len(received) > 1 else 0,
                "Time Diff between first and last (Mins)": (df_eth["timeStamp"].max() - df_eth["timeStamp"].min()).total_seconds() / 60 if len(df_eth) > 1 else 0,
                "Sent tnx": len(sent),
                "Received Tnx": len(received),
                "Unique Received From Addresses": received["from"].nunique() if not received.empty else 0,
                "Unique Sent To Addresses": sent["to"].nunique() if not sent.empty else 0,
                "min value received": received["eth_value"].min() if not received.empty else 0,
                "max value received": received["eth_value"].max() if not received.empty else 0,
                "avg val received": received["eth_value"].mean() if not received.empty else 0,
                "min val sent": sent["eth_value"].min() if not sent.empty else 0,
                "max val sent": sent["eth_value"].max() if not sent.empty else 0,
                "avg val sent": sent["eth_value"].mean() if not sent.empty else 0,
                "total transactions (including tnx to create contract": len(df_eth),
                "total Ether sent": sent["eth_value"].sum(),
                "total ether received": received["eth_value"].sum(),
                "total ether balance": received["eth_value"].sum() - sent["eth_value"].sum()
            }

            new_row = pd.DataFrame([features])
            st.session_state.wallet_dataset = pd.concat(
                [new_row, st.session_state.wallet_dataset], ignore_index=True
            )

        st.success(f"✅ Adresse {address_input} calculée à partir d'Etherscan et ajoutée avec succès au dataset.")

    if not st.session_state.wallet_dataset.empty:
        st.subheader("📋 Aperçu du dataset construit")

        if st.button("🧹 Supprimer les doublons par adresse", use_container_width=True):
            avant = len(st.session_state.wallet_dataset)
            st.session_state.wallet_dataset = st.session_state.wallet_dataset.drop_duplicates(subset="Address")
            après = len(st.session_state.wallet_dataset)
            st.success(f"✅ {avant - après} doublon(s) supprimé(s) !")

        colonnes = st.session_state.wallet_dataset.columns.tolist()
        if "Address" in colonnes:
            colonnes = ["Address"] + [col for col in colonnes if col != "Address"]
            df_affichage = st.session_state.wallet_dataset[colonnes]
        else:
            df_affichage = st.session_state.wallet_dataset.copy()

        st.dataframe(df_affichage)

        st.download_button(
            label="📥 Télécharger le dataset généré (CSV)",
            data=df_affichage.to_csv(index=False).encode("utf-8"),
            file_name="dataset_blockchain.csv",
            mime="text/csv"
        )

############################################################################################################################################################
# 📝 À propos
elif page == "📝 À propos":
    st.title("📝 À propos")

    st.markdown("""
    **EtherScam** est une application open-source conçue pour **analyser le comportement d’un wallet Ethereum** à partir de ses transactions.

    Elle combine :
    - une **analyse comportementale** fondée sur les patterns classiques de fraude
    - une **intelligence artificielle** pour détecter automatiquement les comportements suspects
    - un **générateur de dataset** pour analyser plusieurs adresses à grande échelle
    - des **visualisations claires** pour appuyer le diagnostic

    Elle s'adresse aux **analystes**, **développeurs**, **journalistes spécialisés**, **victimes de fraude**, ou tout simplement aux **curieux** qui souhaitent **vérifier une adresse avant d’interagir avec**.

    ---

    ### 🕵️ Signaler un wallet suspect

    Si une adresse présente un comportement frauduleux, tu peux la signaler via ces plateformes reconnues :

    - [CryptoScamDB](https://cryptoscamdb.org/) : base communautaire recensant les arnaques connues  
    - [Etherscan Report](https://etherscan.io/report) : formulaire officiel pour signaler une adresse  
    - [Chainabuse](https://www.chainabuse.com/) : plateforme collective soutenue par Coinbase, Binance et d'autres acteurs  
    - [Pharos](https://www.internet-signalement.gouv.fr/) : site officiel du gouvernement français pour signaler une fraude en ligne  
    - [Interpol Cybercrime](https://www.interpol.int/en/Crimes/Cybercrime) : contact international pour cybercriminalité

    Ces outils permettent de **renforcer la sécurité de l’écosystème crypto** en identifiant les wallets malveillants.

    ---

    Cette application est **gratuite**, **sans collecte de données**, et maintenue de manière indépendante.  
    Elle continue d’évoluer grâce à vos retours.
    """)

############################################################################################################################################################
# 🤖 Prédiction IA
elif page == "🤖 Prédiction IA":
    st.title("🤖 Prédiction IA")

    st.markdown("""
    Ici, tu peux **tester l’adresse via un modèle d’intelligence artificielle entraîné sur des milliers de wallets**.

    Deux options :
    - Charger un **fichier CSV avec plusieurs adresses** (peut être généré automatiquement via la rubrique : [⚙️ Générateur de Données]
    - Tester une **adresse unique** directement depuis l’interface

    Le modèle IA retourne :
    - un **flag “scam ou non”**
    - une **probabilité associée au scam**
    - et des **graphiques pour évaluer la performance du modèle** (matrice de confusion, importance des variables…)
    """)

    # 🔁 Fonction de normalisation des noms de colonnes
    def normaliser_colonne(nom):
        return unicodedata.normalize('NFKD', nom).encode('ascii', 'ignore').decode().lower() \
            .replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")

    # ✅ Colonnes attendues par le modèle
    colonnes_features = [
        "Address",
        "Avg min between sent tnx",
        "Avg min between received tnx",
        "Time Diff between first and last (Mins)",
        "Sent tnx",
        "Received Tnx",
        "Unique Received From Addresses",
        "Unique Sent To Addresses",
        "min value received",
        "max value received",
        "avg val received",
        "min val sent",
        "max val sent",
        "avg val sent",
        "total transactions (including tnx to create contract",
        "total Ether sent",
        "total ether received",
        "total ether balance"
    ]
    colonnes_features_norm = [normaliser_colonne(c) for c in colonnes_features]

    st.write("Vous pouvez vous servir de la rubrique générateur de dataset, le fichier prends en compte la casse mais doit contenir les colonnes suivantes dans cet ordre :")
    st.code(", ".join(colonnes_features))
    # ⚙️ Charger le modèle
    try:
        model = joblib.load("models/xgb_best_model.pkl")
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle : {e}")
        st.stop()

    # 📤 Import du fichier CSV
    uploaded_file = st.file_uploader("📥 Importer un fichier CSV à prédire :", type=["csv"])

    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file)

            # 🧼 Supprimer les colonnes existantes 'FLAG' ou 'Probabilité Scam (%)' si présentes
            df_raw.drop(columns=[col for col in df_raw.columns if col.strip().lower() in ["flag", "probabilité scam (%)"]],
                        inplace=True, errors="ignore")

            # 🔁 Normalisation et mapping
            df_columns_norm = [normaliser_colonne(col) for col in df_raw.columns]
            mapping = {}
            for col_attendue_norm, col_attendue in zip(colonnes_features_norm, colonnes_features):
                match = difflib.get_close_matches(col_attendue_norm, df_columns_norm, n=1, cutoff=0.85)
                if match:
                    colonne_trouvee = df_raw.columns[df_columns_norm.index(match[0])]
                    mapping[colonne_trouvee] = col_attendue

            df_renamed = df_raw.rename(columns=mapping)

            # ⏳ Aperçu juste après renommage
            st.subheader("🔁 Données juste après renommage intelligent")
            st.dataframe(df_renamed.head())

            # 🔍 Vérifier colonnes manquantes
            colonnes_manquantes = [col for col in colonnes_features if col not in df_renamed.columns]
            if colonnes_manquantes:
                st.error("❌ Colonnes manquantes ou incorrectes :")
                st.code(colonnes_manquantes)
                st.info("Colonnes présentes :")
                st.code(df_renamed.columns.tolist())
                st.stop()

            # ✅ Prédictions
            X_input = df_renamed.drop(columns=["Address"]).fillna(0)
            df_renamed["FLAG"] = model.predict(X_input)
            df_renamed["Probabilité Scam (%)"] = (model.predict_proba(X_input)[:, 1] * 100).round(2)

            # 👁️‍🗨️ Aperçu après prédiction
            st.subheader("🤖 Données après prédictions IA")
            st.dataframe(df_renamed.head())

            # 📊 Statistiques sur les prédictions
            total = len(df_renamed)
            nb_flags = df_renamed["FLAG"].sum()
            pourcentage = (nb_flags / total) * 100

            # 🔢 KPIs
            col1, col2, col3 = st.columns(3)
            col1.metric("📈 Lignes analysées", f"{total}")
            col2.metric("🚨 Proportion FLAG = 1", f"{pourcentage:.2f}%")
            col3.metric("✅ Fiabilité du modèle", "95.63%")

            # 📥 Téléchargement
            st.download_button(
                label="📥 Télécharger les résultats complets avec prédictions",
                data=df_renamed.to_csv(index=False).encode("utf-8"),
                file_name="wallets_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"⚠️ Erreur lors du traitement du fichier : {e}")

##################################################################################################################################
    # 📥 Adresse à tester
    address = st.text_input("🔎 Adresse Ethereum à prédire :", "0xD0cC2B24980CBCCA47EF755Da88B220a82291407")
    api_key = "GAK4SSJCDJDURKJMB8RM62QDW84HJZT57T"

    # ⚙️ Chargement du modèle
    try:
        model = joblib.load("models/xgb_best_model.pkl")
    except FileNotFoundError:
        st.error("❌ Modèle IA introuvable. Vérifie que 'models/xgb_best_model.pkl' existe.")
        st.stop()

    # 🔘 Boutons
    col_analyse, col_reset = st.columns([3, 1])

    if "triggered" not in st.session_state:
        st.session_state.triggered = False

    if col_analyse.button("🚀 Lancer l'analyse complète", use_container_width=True):
        st.session_state.triggered = True

    if col_reset.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.triggered = False
        st.rerun()

    # 🧠 Analyse si déclenchée
    if st.session_state.triggered:
        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={api_key}"
        response = requests.get(url)
        data = response.json()

        if data["status"] != "1":
            st.error("❌ Adresse invalide ou aucune transaction trouvée.")
        else:
            st.success("✅ Analyse effectuée !")
            df = pd.DataFrame(data["result"])
            df["datetime"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s")
            df["timeStamp"] = df["datetime"]
            df["eth_value"] = df["value"].astype(float) / 1e18

            sent = df[df["from"].str.lower() == address.lower()].sort_values("timeStamp")
            received = df[df["to"].str.lower() == address.lower()].sort_values("timeStamp")
            df_eth = pd.concat([sent, received]).sort_values("timeStamp")

            # 🎯 Calcul des features
            features = {
                "Avg min between sent tnx": round(sent["timeStamp"].diff().dt.total_seconds().div(60).mean(), 2) if len(sent) > 1 else np.nan,
                "Avg min between received tnx": round(received["timeStamp"].diff().dt.total_seconds().div(60).mean(), 2) if len(received) > 1 else np.nan,
                "Time Diff between first and last (Mins)": round((df_eth["timeStamp"].max() - df_eth["timeStamp"].min()).total_seconds() / 60, 2) if len(df_eth) > 1 else np.nan,
                "Sent tnx": int(len(sent)),
                "Received Tnx": int(len(received)),
                "Unique Received From Addresses": int(received["from"].nunique()) if not received.empty else np.nan,
                "Unique Sent To Addresses": int(sent["to"].nunique()) if not sent.empty else np.nan,
                "min value received": round(received["eth_value"].min(), 4) if not received.empty else np.nan,
                "max value received": round(received["eth_value"].max(), 4) if not received.empty else np.nan,
                "avg val received": round(received["eth_value"].mean(), 4) if not received.empty else np.nan,
                "min val sent": round(sent["eth_value"].min(), 4) if not sent.empty else np.nan,
                "max val sent": round(sent["eth_value"].max(), 4) if not sent.empty else np.nan,
                "avg val sent": round(sent["eth_value"].mean(), 4) if not sent.empty else np.nan,
                "total transactions (including tnx to create contract": len(df_eth),
                "total Ether sent": round(sent["eth_value"].sum(), 4),
                "total ether received": round(received["eth_value"].sum(), 4),
                "total ether balance": round(received["eth_value"].sum() - sent["eth_value"].sum(), 4)
            }

            # 📈 Prédiction IA
            X_input = pd.DataFrame([features])[[
                "Avg min between sent tnx",
                "Avg min between received tnx",
                "Time Diff between first and last (Mins)",
                "Sent tnx",
                "Received Tnx",
                "Unique Received From Addresses",
                "Unique Sent To Addresses",
                "min value received",
                "max value received",
                "avg val received",
                "min val sent",
                "max val sent",
                "avg val sent",
                "total transactions (including tnx to create contract",
                "total Ether sent",
                "total ether received",
                "total ether balance"
            ]].fillna(0)

            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1]

            # 🎯 Seuil de classification = 0.5
            seuil = 0.5
            if prediction == 1:
                st.error(f"""
                🚨 **Alerte Scam Détecté !**

                Ce wallet présente un **risque élevé de comportement frauduleux**.

                > 🧠 **Confiance du wallet** : `{proba * 100:.2f}%` de chance que ce soit un scam  
                > ⚖️ Seuil de classification : > `{seuil * 100:.0f}%`
                
                Le modèle a analysé ses caractéristiques et estime **avec un très haut niveau de certitude** que ce portefeuille est **potentiellement malveillant**.

                **⚠️ Attention recommandée avant toute interaction et transaction avec le détenteur de cette adresse.**
                """)
            else:
                st.success(f"""
                ✅ **Wallet considéré comme normal**

                Aucune anomalie détectée par l’IA sur ce wallet.

                > 🧠 **Confiance du wallet** : `{(1 - proba) * 100:.2f}%` que ce ne soit **pas** un scam  
                > ⚖️ Seuil de classification : < `{seuil * 100:.0f}%`
                
                D’après les transactions observées, ce wallet présente un comportement classique et **ne déclenche aucun signal fort de scam**.

                **🟢 Pas de suspicion à ce stade.**
                """)

            st.markdown("## 📊 Résumé visuel du modèle")

            # Dossier contenant les images
            image_folder = "images"

            # Groupes d’images : [(titre, nom_fichier)]
            group1 = [("", "best_model_summary.png"), ("", "accuracy_gauge.png")]
            group2 = [("", "classification_report_heatmap.png"), ("", "confusion_matrix.png")]

            # Fonction d'affichage d'un groupe sur une ligne
            def afficher_groupe(group, group_title, explications):
                st.markdown(f"### {group_title}")
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                for i, (ax, (title, filename)) in enumerate(zip(axes, group)):
                    path = os.path.join(image_folder, filename)
                    if os.path.exists(path):
                        img = Image.open(path)
                        ax.imshow(img)
                        ax.set_title(title, fontsize=11)
                        ax.axis("off")
                    else:
                        ax.text(0.5, 0.5, "Image non trouvée", ha='center', va='center', fontsize=12)
                        ax.axis("off")
                st.pyplot(fig)

                # === EXPLICATIONS
                for titre, commentaire in explications.items():
                    st.markdown(f"**{titre}** : {commentaire}")

            # === AFFICHAGE GROUPE 1
            explications1 = {
                "- Résumé": "Ce résumé présente les **meilleurs paramètres** choisis automatiquement pour entraîner le modèle XGBoost à l'aide d'un GridSearch. On observe une **précision de 95,63%**, ce qui signifie que 95 transactions sur 100 sont bien classifiées. Le taux d’erreur est inférieur à 5%, ce qui est un excellent score en classification.",
                "- Jauge": "La jauge donne une représentation **visuelle immédiate de la précision du modèle**, ici à **96%**, ce qui est remarquable. Cette précision élevée montre que le modèle est **très performant pour distinguer un wallet Ethereum suspect d’un wallet légitime**, avec très peu de fausses alertes."
            }
            afficher_groupe(group1, "Meilleur Modèle & Précision", explications1)

            # === AFFICHAGE GROUPE 2
            explications2 = {
                "- Heatmap": "Cette heatmap affiche les **scores de précision, rappel et F1-score** pour chaque classe. La classe 0 (wallet normal) a un rappel de 98%, la classe 1 (scam) un rappel de 84%. Le F1-score moyen dépasse **92%**, ce qui montre l’équilibre du modèle.",
                "- Confusion": "La matrice montre une **faible erreur de classification** : peu de faux positifs (28) et faux négatifs (53). Le modèle **identifie bien les scams tout en évitant les fausses alertes**, ce qui garantit sa fiabilité."
            }
            afficher_groupe(group2, "Performance du Modèle", explications2)



            # === IMPORTANCE DES VARIABLES
            img_path = os.path.join(image_folder, "feature_importance.png")
            if os.path.exists(img_path):
                st.markdown("### Importance des variables")
                st.image(Image.open(img_path), use_container_width=True)

                st.markdown("""
            **Analyse** : Ce graphique montre les **variables les plus influentes** pour détecter un scam.

            - `total ether received` et `Unique Received From Addresses` sont très prédictives : les scams reçoivent souvent **beaucoup d’argent de plusieurs sources différentes**.
            - Les variables de **fréquence** (`Avg min between sent/received tnx`) et de **durée d'activité** (`Time Diff`) sont aussi essentielles pour repérer les comportements inhabituels.
            - Le **solde final**, les **valeurs moyennes envoyées/reçues**, et le **nombre total de transactions** complètent la lecture du modèle.

            ✅ Notre modèle s’appuie donc sur des critères concrets, comportementaux et monétaires.
            """)
            else:
                st.warning("❌ Image 'feature_importance.png' non trouvée.")

############################################################################################################################################################
# 🚨 Analyse Wallet (fusion analyse + détection)
elif page == "🚨 Analyse Wallet":
    st.title("🚨 Analyse Wallet")

    st.markdown("""
    Dans cette section, tu peux **vérifier une adresse Ethereum** pour détecter des comportements suspects.  
    L’analyse combine :
    - des **indicateurs clés (KPIs)** : nombre de transactions, volume total, solde…
    - une **analyse comportementale complète** : temporalité, ratios, patterns types (burner, scam…)
    - des **visualisations interactives** pour comprendre les flux du wallet.

    L’objectif est de **détecter rapidement un comportement frauduleux** à partir d’une simple adresse.
    """)


    address = st.text_input("🔎 Adresse Ethereum à analyser :", "0xD0cC2B24980CBCCA47EF755Da88B220a82291407")
    api_key = "GAK4SSJCDJDURKJMB8RM62QDW84HJZT57T"

    # Colonne pour boutons alignés
    col_analyse, col_reset = st.columns([3, 1])

    if "triggered" not in st.session_state:
        st.session_state.triggered = False

    if col_analyse.button("🚀 Lancer l'analyse complète", use_container_width=True):
        st.session_state.triggered = True

    if col_reset.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.triggered = False
        st.rerun()


    # On lance l'analyse seulement si triggered = True
    if st.session_state.triggered:

        url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={api_key}"
        with st.spinner("📡 Connexion à Etherscan..."):
            response = requests.get(url)
            data = response.json()
        
        if data["status"] != "1":
            st.error("❌ Adresse invalide ou aucune transaction trouvée sur Etherscan.")

            st.info(f"""
            🔎 Cette application permet d’analyser si un **wallet Ethereum est potentiellement suspect** ou classifié comme scam.

            🔗 Voir sur [CryptoScamDB.org](https://cryptoscamdb.org/scams)  
            🔗 Voir l’adresse sur [Etherscan.io](https://etherscan.io/address/{address})
            """)
        else:
            st.success("✅ Analyse effectuée !")
            
            st.info(f"""
            🔎 Cette application permet d’analyser si un **wallet Ethereum est potentiellement suspect** ou référencé comme scam.

            - Voir les scams référencés sur : [CryptoScamDB.org](https://cryptoscamdb.org/scams)  
            - Voir les transactions de l’adresse sur : [Etherscan.io](https://etherscan.io/address/{address})
            """)

            df = pd.DataFrame(data["result"])
            df["datetime"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s")
            df["timeStamp"] = df["datetime"]
            df["eth_value"] = df["value"].astype(float) / 1e18

            addr_lower = address.lower()
            sent = df[df["from"].str.lower() == addr_lower].sort_values("timeStamp")
            received = df[df["to"].str.lower() == addr_lower].sort_values("timeStamp")
            df_eth = pd.concat([sent, received]).sort_values("timeStamp")

            lifetime_days = (df_eth["timeStamp"].max() - df_eth["timeStamp"].min()).days
            

            # Résumé analytique — à placer AVANT toute utilisation de la variable 'features'
            features = {
                "Address": address,
                "Sent tnx": len(sent),
                "Received Tnx": len(received),
                "Unique Received From Addresses": received["from"].nunique() if not received.empty else np.nan,
                "Unique Sent To Addresses": sent["to"].nunique() if not sent.empty else np.nan,
                "min value received": received["eth_value"].min() if not received.empty else np.nan,
                "max value received": received["eth_value"].max() if not received.empty else np.nan,
                "avg val received": received["eth_value"].mean() if not received.empty else np.nan,
                "min val sent": sent["eth_value"].min() if not sent.empty else np.nan,
                "max val sent": sent["eth_value"].max() if not sent.empty else np.nan,
                "avg val sent": sent["eth_value"].mean() if not sent.empty else np.nan,
                "total Ether sent": sent["eth_value"].sum(),
                "total ether received": received["eth_value"].sum(),
                "total ether balance": received["eth_value"].sum() - sent["eth_value"].sum()
            }

            danger_score, diagnostics, profile_tags = analyse_wallet_complete(features, df_eth, lifetime_days, address)

            confidence_score = 10 - danger_score

            if confidence_score <= 4:
                st.error(f"💀 Indice de confiance : {confidence_score}/10 — **Ce wallet présente un profil de risque EXTRÊME. Fuyez.**")
            elif confidence_score <= 6:
                st.warning(f"⚠️ Indice de confiance : {confidence_score}/10 — **Multiples signaux d’alerte. Restez méfiant.**")
            elif confidence_score <= 8:
                st.info(f"🧐 Indice de confiance : {confidence_score}/10 — **Anomalies détectées. Vigilance conseillée.**")
            else:
                st.success(f"🟢 Indice de confiance : {confidence_score}/10 — Aucun comportement frauduleux évident détecté.")

            def get_progress_color(score):
                # Rouge (0) → Orange (5) → Vert (10)
                if score <= 4:
                    return "#e74c3c"  # Rouge
                elif score <= 6:
                    return "#f39c12"  # Orange
                elif score <= 8:
                    return "#f1c40f"  # Jaune
                else:
                    return "#2ecc71"  # Vert

            color = get_progress_color(confidence_score)

            st.markdown(f"""
            <div style="background-color: #e0e0e0; border-radius: 10px; height: 22px; width: 100%;">
            <div style="background-color: {color}; width: {confidence_score * 10}%; height: 100%; border-radius: 10px;"></div>
            </div>
            """, unsafe_allow_html=True)


            # KPIs en colonnes
            st.markdown(" ")  
            st.subheader("📌 KPIs")
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Solde final", f"{features['total ether balance']:.4f} ETH")
            col2.metric("📤 Total envoyé", f"{features['total Ether sent']:.2f} ETH")
            col3.metric("📥 Total reçu", f"{features['total ether received']:.2f} ETH")

            col4, col5, col6 = st.columns(3)
            col4.metric("⚙️ Tx envoyées", features["Sent tnx"])
            col5.metric("📬 Tx reçues", features["Received Tnx"])
            col6.metric("⏳ Durée d'activité", f"{lifetime_days} jours")

            # Tableau
            # ✅ Reconstruction complète des features avec toutes les métriques du script terminal
            # ✅ Reconstruction complète des features avec toutes les métriques du script terminal
            features = {
                "Address": address,
                "FLAG": np.nan,
                "Avg min between sent tnx": sent["timeStamp"].diff().dt.total_seconds().div(60).mean() if len(sent) > 1 else np.nan,
                "Avg min between received tnx": received["timeStamp"].diff().dt.total_seconds().div(60).mean() if len(received) > 1 else np.nan,
                "Time Diff between first and last (Mins)": (df_eth["timeStamp"].max() - df_eth["timeStamp"].min()).total_seconds() / 60 if len(df_eth) > 1 else np.nan,
                "Sent tnx": len(sent),
                "Received Tnx": len(received),
                "Unique Received From Addresses": received["from"].nunique() if not received.empty else np.nan,
                "Unique Sent To Addresses": sent["to"].nunique() if not sent.empty else np.nan,
                "min value received": received["eth_value"].min() if not received.empty else np.nan,
                "max value received ": received["eth_value"].max() if not received.empty else np.nan,  # garde bien l'espace
                "avg val received": received["eth_value"].mean() if not received.empty else np.nan,
                "min val sent": sent["eth_value"].min() if not sent.empty else np.nan,
                "max val sent": sent["eth_value"].max() if not sent.empty else np.nan,
                "avg val sent": sent["eth_value"].mean() if not sent.empty else np.nan,
                "total transactions (including tnx to create contract": len(df_eth),
                "total Ether sent": sent["eth_value"].sum(),
                "total ether received": received["eth_value"].sum(),
                "total ether balance": received["eth_value"].sum() - sent["eth_value"].sum()
            }

            # 📊 Conversion en DataFrame
            kpi_data = pd.DataFrame.from_dict(features, orient="index", columns=["Valeur"]).reset_index()
            kpi_data.columns = ["KPI", "Valeur"]

            # 🎯 Format intelligent selon le KPI
            def format_value(kpi, val):
                if pd.isna(val):
                    return ""
                if kpi in ["Address", "FLAG"]:
                    return val
                elif "avg min between" in kpi.lower() or "time diff" in kpi.lower():
                    return f"{val:.2f} min"
                elif "tnx" in kpi.lower() or "unique" in kpi.lower():
                    return f"{int(val)}"
                elif "value" in kpi.lower() or "val " in kpi.lower() or "balance" in kpi.lower() or "total ether" in kpi.lower():
                    return f"{val:.4f} ETH"
                else:
                    return val

            kpi_data["Valeur"] = kpi_data.apply(lambda row: format_value(row["KPI"], row["Valeur"]), axis=1)

            # 📋 Affichage final du tableau
            st.dataframe(kpi_data.set_index("KPI"), use_container_width=True)

##############################################################################################################
# 🔍 Analyse ultime : comportement, finance, temporalité, classification
# 🔍 Analyse ultra-détaillée : comportement, finance, temporalité, classification avancée# 🔍 Analyse ultra-détaillée : comportement, finance, temporalité, classification avancée# 🔍 Analyse ultra-détaillée : comportement, finance, temporalité, classification avancée
            st.markdown(" ")  
            st.subheader("🧠 Analyses")

            diagnostics = []
            profile_tags = []

            # Variables de base
            sent_count = features["Sent tnx"]
            recv_count = features["Received Tnx"]
            sent_sum = features["total Ether sent"]
            recv_sum = features["total ether received"]
            balance = features["total ether balance"]
            sent_to = features["Unique Sent To Addresses"]
            recv_from = features["Unique Received From Addresses"]
            redistribution_ratio = 1 - (balance / (recv_sum + 1e-6))
            avg_val_sent = features["avg val sent"]
            avg_val_received = features["avg val received"]
            tx_ratio = sent_count / (recv_count + 1e-6)

            # Analyse temporelle
            if lifetime_days < 15:
                diagnostics.append(f"- Wallet actif depuis {lifetime_days} jours seulement. Très jeune, typique des scams temporaires.")
            elif lifetime_days < 90:
                diagnostics.append(f"- Wallet jeune (≈ {lifetime_days} jours). Historique limité, attention au contexte.")
            elif lifetime_days < 365:
                diagnostics.append(f"- Wallet actif depuis moins d’un an : {lifetime_days} jours. Antériorité moyenne.")
            else:
                diagnostics.append(f"- Wallet actif depuis {lifetime_days} jours. Ancienneté rassurante.")
                if sent_sum > 300 and recv_sum > 300:
                    diagnostics.append("- Forte activité financière sur longue période. Le wallet gère des volumes importants.")
                elif sent_sum > 300 and balance < 1:
                    diagnostics.append("- Wallet très actif et ancien, mais totalement vidé. Dump, blanchiment ou fuite de capitaux ?")
                elif recv_sum > 300 and sent_sum < 1:
                    diagnostics.append("- Wallet ancien qui a reçu beaucoup sans rien envoyer. Potentiel piège à fonds ou cold storage.")

            # Activité par heure et jour
            df_eth["hour"] = df_eth["timeStamp"].dt.hour
            df_eth["weekday"] = df_eth["timeStamp"].dt.dayofweek
            most_active_hour = df_eth["hour"].mode()[0]
            most_active_day = df_eth["weekday"].mode()[0]
            hour_freq = df_eth["hour"].value_counts(normalize=True).max()

            if hour_freq > 0.5:
                diagnostics.append(f"- Plus de 50% des transactions ont lieu à {most_active_hour}h. Probable script ou automatisation.")
            else:
                diagnostics.append(f"- Activité répartie : heure dominante = {most_active_hour}h.")

            diagnostics.append(f"- Jour dominant = {['Lun','Mar','Mer','Jeu','Ven','Sam','Dim'][most_active_day]}.")

            # Ratio tx
            if recv_count == 0 and sent_count > 0:
                diagnostics.append("- Le wallet n’a jamais reçu d’ETH, mais envoie. Relais ? Burner ?")
            elif sent_count == 0 and recv_count > 0:
                diagnostics.append("- Le wallet ne fait que recevoir. Peut attirer des fonds ?")
            else:
                if tx_ratio > 10:
                    diagnostics.append("- Ratio tx >10 : envoie beaucoup plus qu’il ne reçoit.")
                elif tx_ratio < 0.1:
                    diagnostics.append("- Ratio tx <0.1 : reçoit beaucoup, envoie très peu.")
                elif tx_ratio > 3:
                    diagnostics.append(f"- Ratio d’envoi élevé : {tx_ratio:.2f}x plus d’envois que de réceptions.")
                else:
                    diagnostics.append("- Ratio envois/réceptions équilibré.")

            # Solde
            if balance < 0.0001 and recv_sum > 1:
                diagnostics.append("- Solde final nul malgré de grosses réceptions. Wallet vidé.")
            elif balance < 0.01:
                diagnostics.append("- Solde très faible. Wallet temporaire ?")
            else:
                if redistribution_ratio > 0.95:
                    diagnostics.append(f"- Redistribution importante : {redistribution_ratio * 100:.2f}% des fonds reçus sont sortis.")
                elif redistribution_ratio < 0.1:
                    diagnostics.append("- Ce wallet conserve la quasi-totalité de ses fonds.")

            # Fragmentation
            if avg_val_sent < avg_val_received * 0.5:
                diagnostics.append(f"- Envois fragmentés : moyenne = {avg_val_sent:.2f} ETH contre {avg_val_received:.2f} ETH reçus.")

            # Grosse transaction
            if features["max value received "] > 250:
                diagnostics.append("- Transaction >250 ETH détectée. Gros flux potentiel.")

            # Patterns comportementaux
            if sent_count > 50 and recv_count < 10:
                diagnostics.append("- Pattern flooder : beaucoup d’envois, peu de réceptions.")
                profile_tags.append("Flooder")

            if sent_to > 50 and balance < 1 and lifetime_days < 90:
                diagnostics.append("- Burner détecté : jeune, balance vide, envoie à >50 adresses.")
                profile_tags.append("Burner")

            if recv_from > 30 and sent_count == 0:
                diagnostics.append("- Collector détecté : reçoit beaucoup, n'envoie rien.")
                profile_tags.append("Collector")

            if recv_sum > 100 and sent_sum < 1:
                diagnostics.append("- A reçu >100 ETH sans rien renvoyer. Comportement scam-like.")
                profile_tags.append("Scam-like")

            # Écart-type
            val_std = df_eth["eth_value"].std()
            if val_std > 10:
                diagnostics.append(f"- Volatilité élevée : écart-type = {val_std:.2f} ETH.")

            # Faible activité
            if (recv_count + sent_count) < 5:
                diagnostics.append("- Moins de 5 transactions. Wallet dormant ou test.")
                profile_tags.append("Dormant")

            # Tx nulles
            if features["min val sent"] == 0.0 or features["min value received"] == 0.0:
                diagnostics.append("- Transactions nulles (0 ETH) détectées.")

            # Affichage des tags et diagnostics
            if profile_tags:
                st.markdown(f"**Profil comportemental détecté :** `{', '.join(profile_tags)}`")

            for diag in diagnostics:
                st.write(diag)


################################################################################################################################################
# 📈 Évolution du solde dans le temps

            # Calcul du solde net
            df_eth["balance_change"] = df_eth.apply(
                lambda row: row["eth_value"] if row["to"].lower() == address.lower() else -row["eth_value"], axis=1
            )
            df_eth = df_eth.sort_values("timeStamp")  # Important pour que le cumsum ait du sens
            df_eth["wallet_balance"] = df_eth["balance_change"].cumsum()

            st.markdown(" ")  
            st.subheader("📈 Visualisations")

            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

            # ✅ 1. Solde cumulé
            ax1.plot(df_eth["timeStamp"], df_eth["wallet_balance"], color="#2980B9", linewidth=2.5, marker='o', markersize=3)
            ax1.set_ylabel("Balance (ETH)")
            ax1.set_title("Wallet Net Cumulative Balance")
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.tick_params(axis='x', rotation=45)

            # ✅ 2. Transactions (réception / envoi)
            ax2.scatter(
                df_eth[df_eth["to"].str.lower() == address.lower()]["timeStamp"],
                df_eth[df_eth["to"].str.lower() == address.lower()]["eth_value"],
                color="#2ECC71", label="Incoming", alpha=0.7
            )

            ax2.scatter(
                df_eth[df_eth["from"].str.lower() == address.lower()]["timeStamp"],
                -df_eth[df_eth["from"].str.lower() == address.lower()]["eth_value"],
                color="#E74C3C", label="Outgoing", alpha=0.7
            )

            ax2.axhline(0, color='black', linewidth=0.8)
            ax2.set_title("Transactions: Inflows (↑) vs Outflows (↓)")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Amount (ETH)")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, linestyle="--", alpha=0.5)
            ax2.legend()

            # Affichage
            plt.tight_layout()
            st.pyplot(fig)

            # 🔥 Recalcul de la carte de chaleur si pas encore fait
            df_eth["hour"] = df_eth["timeStamp"].dt.hour
            df_eth["day"] = df_eth["timeStamp"].dt.dayofweek
            heatmap_data = df_eth.groupby(["day", "hour"]).size().unstack().fillna(0)

            # Affichage
            fig3, ax3 = plt.subplots(figsize=(10, 5))


            sns.heatmap(
                heatmap_data,
                cmap="YlOrRd",
                linewidths=0.5,
                annot=True,
                fmt=".0f",
                cbar_kws={"label": "Nb de transactions"},
                ax=ax3
            )
            ax3.set_title("Transaction Count by Hour and Day")
            ax3.set_xlabel("Hour of Day")
            ax3.set_ylabel("Day of Week")
            day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

            visible_days = heatmap_data.index.tolist()
            ax3.set_yticklabels([day_labels[d] for d in visible_days], rotation=0)

            st.pyplot(fig3)

            # Données pour camembert (nombre de transactions)
            # Data for pie chart (number of transactions)
            tx_counts = [len(sent), len(received)]
            tx_labels = ["Sent", "Received"]
            tx_colors = ["#E74C3C", "#2ECC71"]

            # Data for treemap (ETH amounts)
            valeurs = [sent["eth_value"].sum(), received["eth_value"].sum()]
            etiquettes = [f"Sent: {valeurs[0]:.2f} ETH", f"Received: {valeurs[1]:.2f} ETH"]
            treemap_colors = ["#E74C3C", "#2ECC71"]


            # 🔎 Filtrage des valeurs nulles ou très faibles pour éviter les erreurs de squarify
            valeurs_filtrees = [v for v in valeurs if v > 0]
            etiquettes_filtrees = [e for v, e in zip(valeurs, etiquettes) if v > 0]
            couleurs_filtrees = [c for v, c in zip(valeurs, treemap_colors) if v > 0]

            # Création des sous-graphiques côte à côte
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 🥧 Pie Chart
            ax1.pie(tx_counts, labels=tx_labels, colors=tx_colors, autopct="%1.1f%%", startangle=140)
            ax1.set_title("Transaction Distribution")

            # 🌳 Treemap ou message d'indisponibilité
            if valeurs_filtrees:
                squarify.plot(sizes=valeurs_filtrees, label=etiquettes_filtrees, color=couleurs_filtrees, alpha=0.8, ax=ax2)
                ax2.set_title("Total Amounts Sent / Received")
            else:
                ax2.text(0.5, 0.5, "Pas assez de volume pour\nafficher un treemap", ha='center', va='center', fontsize=12)
                ax2.set_title("Treemap Unavailable")

            ax2.axis("off")

            # 🖼️ Affichage
            st.pyplot(fig)

            st.markdown("""
            **Solde cumulé (Wallet Net Cumulative Balance)** :  
            Ce graphique montre l’évolution du solde en ETH du wallet dans le temps.  
            - Une hausse brutale en escalier sans variations liés à des investissements suivie d’un vidage complet peut signaler un usage temporaire, typique des wallets de transit ou de blanchiment.  
            - Un solde stable ou croissant sur le long terme peut indiquer une conservation volontaire des fonds (comportement HODL).

            **Transactions en entrée et sortie** :  
            Les points positifs indiquent les fonds reçus, les négatifs les fonds envoyés.  
            - Des mouvements rapprochés et opposés (réception suivie d’un envoi) peuvent suggérer un relais de fonds.  
            - Un déséquilibre fort entre nombre de transactions reçues et envoyées peut trahir un pattern suspect.

            **Carte de chaleur des jours et heures** :  
            Cette carte indique à quels moments le wallet est le plus actif.  
            - Une concentration sur un créneau précis (ex. tous les jeudis matin) est typique d’une automatisation via script ou bot.  
            - Une activité répartie sur plusieurs jours et heures est plus représentative d’un usage humain naturel.

            **Répartition des transactions (camembert)** :  
            Ce graphique compare le nombre de transactions envoyées et reçues.  
            - Un déséquilibre prononcé peut révéler une stratégie de collecte (ex. uniquement recevoir) ou de dispersion pour blanchir (ex. uniquement envoyer).

            **Répartition des volumes (treemap)** :  
            Ce graphique compare les montants totaux reçus et envoyés.  
            - Un wallet qui reçoit beaucoup d’ETH sans quasiment rien envoyer peut être un piège à fonds.  
            - À l’inverse, un wallet qui vide rapidement ce qu’il reçoit peut correspondre à un portefeuille jetable ou utilisé pour dissimuler l’origine des fonds.
            """)
