from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from demo.data import FEATURE_COLS, load_contacts
from demo.api import score_all, score_one

st.set_page_config(
    page_title="ChurnGuard — SeoLap Demo",
    page_icon="🔮",
    layout="wide",
)

RISK_COLOR = {"high": "🔴", "medium": "🟡", "low": "🟢"}
RISK_LABEL = {"high": "Élevé", "medium": "Moyen", "low": "Faible"}


# ---------- Session state ----------

if "contacts" not in st.session_state:
    st.session_state.contacts = load_contacts()

if "scored" not in st.session_state:
    st.session_state.scored = False


# ---------- Header ----------

st.title("🔮 ChurnGuard — Démo SeoLap")
st.caption("Pipeline MLOps · Jedha Data Science Lead · Emeline Roblot")
st.divider()


# ---------- Tabs ----------

tab_dashboard, tab_client = st.tabs(["📊 Dashboard", "👤 Fiche client"])


# ==============================
# TAB 1 — Dashboard
# ==============================
with tab_dashboard:
    df = st.session_state.contacts

    col_btn, col_info = st.columns([2, 5])
    with col_btn:
        if st.button("⚡ Scorer tous les contacts", type="primary", use_container_width=True):
            with st.spinner("Appel API en cours…"):
                try:
                    results = score_all(df)
                    now = datetime.now(timezone.utc).strftime("%H:%M:%S")
                    for r in results:
                        mask = df["id"] == r["account_id"]
                        df.loc[mask, "churn_score"] = r["churn_score"]
                        df.loc[mask, "churn_risk"] = r["churn_risk"]
                        df.loc[mask, "last_predicted_at"] = now
                    st.session_state.contacts = df
                    st.session_state.scored = True
                    st.success(f"{len(results)} contacts scorés.")
                except Exception as e:
                    st.error(f"Erreur API : {e}")

    with col_info:
        if st.session_state.scored:
            scored_df = df[df["churn_score"].notna()]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total contacts", len(df))
            c2.metric("🔴 Risque élevé", (scored_df["churn_risk"] == "high").sum())
            c3.metric("🟡 Risque moyen", (scored_df["churn_risk"] == "medium").sum())
            c4.metric("🟢 Risque faible", (scored_df["churn_risk"] == "low").sum())

    st.divider()

    # Tableau
    display = df[["email", "Tenure", "Usage Frequency", "Last Interaction", "churn_score", "churn_risk"]].copy()
    display.columns = ["Email", "Ancienneté (mois)", "Logins", "Dernière activité (j)", "Score de churn", "Risque"]

    if st.session_state.scored:
        display = display.sort_values("Score de churn", ascending=False)
        display["Risque"] = display["Risque"].map(
            lambda r: f"{RISK_COLOR.get(r, '')} {RISK_LABEL.get(r, r)}" if pd.notna(r) else "—"
        )
        display["Score de churn"] = display["Score de churn"].apply(
            lambda s: f"{s:.3f}" if pd.notna(s) else "—"
        )
    else:
        display["Score de churn"] = "—"
        display["Risque"] = "—"

    st.dataframe(display, use_container_width=True, hide_index=True)


# ==============================
# TAB 2 — Fiche client
# ==============================
with tab_client:
    df = st.session_state.contacts

    selected_email = st.selectbox(
        "Sélectionner un contact",
        options=df["email"].tolist(),
        index=0,
    )

    contact_idx = df[df["email"] == selected_email].index[0]
    contact = df.loc[contact_idx].copy()

    st.subheader(f"📧 {selected_email}")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Données SeoLap (réelles)**")
        tenure = st.number_input("Ancienneté (mois)", value=float(contact["Tenure"]), min_value=0.0, step=0.5)
        usage = st.number_input("Logins (Usage Frequency)", value=int(contact["Usage Frequency"]), min_value=0, step=1)
        last_int = st.number_input("Jours depuis dernière activité", value=int(contact["Last Interaction"]), min_value=0, step=1)
        contract = st.selectbox("Contrat", ["Monthly (0)", "Quarterly (1)", "Annual (2)"], index=int(contact["Contract Length"]))
        contract_val = int(contract.split("(")[1].rstrip(")"))

    with col_right:
        st.markdown("**Données complétées (modifiables pour la démo)**")
        age = st.number_input("Âge", value=int(contact["Age"]), min_value=18, max_value=80, step=1)
        gender = st.selectbox("Genre", ["Homme (1)", "Femme (0)"], index=0 if contact["Gender"] == 1 else 1)
        gender_val = int(gender.split("(")[1].rstrip(")"))
        support = st.number_input("Appels support", value=int(contact["Support Calls"]), min_value=0, step=1)
        payment_delay = st.number_input("Retard de paiement (jours)", value=int(contact["Payment Delay"]), min_value=0, step=1)
        total_spend = st.number_input("Dépense totale (€)", value=float(contact["Total Spend"]), min_value=0.0, step=10.0)
        sub_type = st.selectbox("Plan", ["Basic", "Standard", "Premium"], index=0)

    # Construire la ligne mise à jour
    updated = contact.copy()
    updated["Tenure"] = tenure
    updated["Usage Frequency"] = usage
    updated["Last Interaction"] = last_int
    updated["Contract Length"] = contract_val
    updated["Age"] = age
    updated["Gender"] = gender_val
    updated["Support Calls"] = support
    updated["Payment Delay"] = payment_delay
    updated["Total Spend"] = total_spend
    updated["Subscription Type_Basic"] = 1 if sub_type == "Basic" else 0
    updated["Subscription Type_Standard"] = 1 if sub_type == "Standard" else 0
    updated["Subscription Type_Premium"] = 1 if sub_type == "Premium" else 0
    updated["support_intensity"] = support / (tenure + 1)
    updated["spend_per_month"] = total_spend / (tenure + 1)
    updated["payment_risk_score"] = payment_delay * support

    st.divider()

    if st.button("🔮 Prédire le churn", type="primary", use_container_width=True):
        with st.spinner("Appel API…"):
            try:
                result = score_one(updated)
                score = result["churn_score"]
                risk = result["churn_risk"]

                # Mettre à jour le session state
                df.loc[contact_idx, "churn_score"] = score
                df.loc[contact_idx, "churn_risk"] = risk
                st.session_state.contacts = df

                # Afficher le résultat
                st.divider()
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Score de churn", f"{score:.3f}")
                rc2.metric("Niveau de risque", f"{RISK_COLOR.get(risk, '')} {RISK_LABEL.get(risk, risk)}")
                rc3.progress(score, text=f"{score*100:.1f}% de probabilité de désabonnement")

                if risk == "high":
                    st.error("⚠️ Risque élevé — action de rétention recommandée.")
                elif risk == "medium":
                    st.warning("Risque modéré — à surveiller.")
                else:
                    st.success("Risque faible — client stable.")

            except Exception as e:
                st.error(f"Erreur API : {e}")
