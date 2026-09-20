import os

import requests

# Space HF par défaut ; en prod AWS le conteneur dashboard reçoit CHURNGUARD_API_URL=http://api:8000
API_URL = os.getenv("CHURNGUARD_API_URL", "https://emeliner-churnguard.hf.space")
TIMEOUT = 30


def score_all(contacts) -> list[dict]:
    """Appelle /predict/batch sur tous les contacts. Retourne la liste des résultats."""
    from demo.data import build_predict_payload

    accounts = [build_predict_payload(row) for _, row in contacts.iterrows()]
    resp = requests.post(
        f"{API_URL}/predict/batch",
        json={"accounts": accounts},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["results"]


def score_one(row) -> dict:
    """Appelle /predict pour un contact unique. Retourne le résultat."""
    from demo.data import build_predict_payload

    resp = requests.post(
        f"{API_URL}/predict",
        json=build_predict_payload(row),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()
