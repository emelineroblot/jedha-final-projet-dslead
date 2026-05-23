---
title: ChurnGuard API
emoji: 🔮
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: MLOps pipeline — Customer churn prediction API (Jedha DSLead)
---

# ChurnGuard API

REST API for customer churn prediction — Jedha Data Science Lead final project.

**Endpoints:**
- `GET /health` — health check
- `POST /predict` — single account churn score
- `POST /predict/batch` — batch scoring
- `GET /model/info` — active model metadata
- `GET /docs` — Swagger UI

See the [GitHub repo](https://github.com/emelineroblot/churnguard) for full documentation.
