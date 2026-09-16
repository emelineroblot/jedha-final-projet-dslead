FROM python:3.11-slim

# libgomp1 : runtime OpenMP requis par XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 churnguard

# Réseau avec inspection TLS (Avast, proxy d'entreprise) : passer --build-arg PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org"
# (vide par défaut → pip vérifie normalement les certificats, comme en CI)
ARG PIP_TRUSTED_HOST=""
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10
WORKDIR /app
COPY requirements-api.txt .
# wheels/ : wheels Linux pré-téléchargés sur l'hôte (réseau lent/intercepté) — vide en CI → PyPI
RUN --mount=type=cache,id=churnguard-pip,target=/root/.cache/pip --mount=type=bind,source=wheels,target=/wheels \
    pip install --find-links /wheels -r requirements-api.txt

COPY --chown=churnguard:churnguard src/ ./src/
USER churnguard

ENV PYTHONUNBUFFERED=1 \
    CHURNGUARD_ROOT=/app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
