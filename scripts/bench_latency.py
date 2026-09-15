"""
Bench de latence de l'API : p50 / p95 / p99 sur /predict (unitaire) et /predict/batch.

    python scripts/bench_latency.py --url http://localhost:8001 --n 200 --batch 1000

Les résultats alimentent la slide "métriques" (objectif : p95 < 200 ms sur /predict).
"""
import argparse
import statistics
import time

import httpx
import pandas as pd

from src.paths import FEATURES_TEST_PATH, TARGET
from src.preprocessing.features import FEATURE_COLUMNS


def _rows(n: int) -> list[dict]:
    df = pd.read_csv(FEATURES_TEST_PATH, nrows=n).drop(columns=[TARGET], errors="ignore")[FEATURE_COLUMNS]
    return [{"account_id": f"bench-{i}", "features": {k: float(v) for k, v in r.items()}} for i, r in enumerate(df.to_dict("records"))]


def _pct(values: list[float]) -> dict:
    values = sorted(values)
    q = lambda p: values[min(int(len(values) * p), len(values) - 1)]  # noqa: E731
    return {"p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "mean": statistics.mean(values), "max": max(values)}


def bench(url: str, n: int, batch: int) -> None:
    client = httpx.Client(base_url=url, timeout=60)
    assert client.get("/health").json()["model_loaded"], "modèle non chargé"
    rows = _rows(max(n, batch))

    single = []
    for row in rows[:n]:
        t = time.perf_counter()
        client.post("/predict", json=row).raise_for_status()
        single.append((time.perf_counter() - t) * 1000)

    batch_times, server_times = [], []
    for _ in range(5):
        t = time.perf_counter()
        r = client.post("/predict/batch", json={"accounts": rows[:batch]})
        r.raise_for_status()
        batch_times.append((time.perf_counter() - t) * 1000)
        server_times.append(r.json()["latency_ms"])

    s = _pct(single)
    print(f"POST /predict        ({n} requêtes)     : p50={s['p50']:.1f} ms  p95={s['p95']:.1f} ms  p99={s['p99']:.1f} ms  max={s['max']:.1f} ms")
    print(f"POST /predict/batch  ({batch} comptes ×5) : total moyen={statistics.mean(batch_times):.0f} ms  "
          f"scoring serveur={statistics.mean(server_times):.0f} ms  → {batch / (statistics.mean(batch_times) / 1000):,.0f} comptes/s")
    print("OBJECTIF p95 < 200 ms :", "OK" if s["p95"] < 200 else "KO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--batch", type=int, default=1000)
    args = parser.parse_args()
    bench(args.url, args.n, args.batch)
