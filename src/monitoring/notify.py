"""
Envoi d'alertes vers un webhook (Discord ou Slack) défini par ALERT_WEBHOOK_URL.

Sans webhook configuré, l'alerte est seulement journalisée — le pipeline ne dépend
jamais de la disponibilité du canal de notification.
"""
import logging
import os

logger = logging.getLogger(__name__)

WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")


def send_alert(title: str, message: str, level: str = "warning") -> bool:
    """Retourne True si l'alerte a été envoyée au webhook, False sinon (log seulement)."""
    icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨", "success": "✅"}.get(level, "⚠️")
    text = f"{icon} **ChurnGuard — {title}**\n{message}"
    getattr(logger, "warning" if level in ("warning", "critical") else "info")("%s — %s", title, message)

    if not WEBHOOK_URL:
        return False
    try:
        import requests

        # `content` = Discord, `text` = Slack ; chaque service ignore la clé de l'autre
        resp = requests.post(WEBHOOK_URL, json={"content": text, "text": text}, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.error("Envoi de l'alerte impossible : %s", exc)
        return False
