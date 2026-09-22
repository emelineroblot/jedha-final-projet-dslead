"""Génère les schémas d'architecture ChurnGuard (SVG) dans docs/architecture/ — style Jedha (blocs colorés + badges outils)."""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "font-family='Inter, Segoe UI, Helvetica, Arial, sans-serif'"
COL = {
    "violet": ("#ECE6FB", "#6B4EE6"), "green": ("#E2F5EA", "#1E9E5A"), "yellow": ("#FFF3D1", "#D69E00"),
    "orange": ("#FFE6D5", "#E8641B"), "cyan": ("#DDF6F6", "#0F9BA8"), "blue": ("#E6EFFD", "#2F6FE4"),
    "grey": ("#EEF0F3", "#7A8391"), "red": ("#FDE2E2", "#D64545"),
}
TOOL_COLORS = {
    "MLflow": "#0194E2", "XGBoost": "#1E88E5", "scikit-learn": "#F7931E", "FastAPI": "#009688", "Docker": "#2496ED",
    "Airflow": "#017CEE", "Evidently": "#E4572E", "PostgreSQL": "#336791", "DVC": "#945DD6", "DagsHub": "#4A2C8F",
    "GitHub Actions": "#24292E", "GHCR": "#24292E", "Streamlit": "#FF4B4B", "Mautic": "#4E5E9E", "Kaggle CSV": "#20BEFF",
    "Prometheus": "#E6522C", "HF Spaces": "#FFB000", "Pydantic": "#E92063", "Discord/Slack": "#5865F2", "pytest": "#0A9EDC",
    "ruff": "#D7FF64", "Python": "#3776AB", "SeoLap": "#111827", "joblib": "#555",
    "AWS": "#FF9900", "Terraform": "#7B42BC", "EC2": "#FF9900", "S3": "#3F8624", "SSM": "#DD344C",
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=13, weight="normal", fill="#1F2937", anchor="start", extra=""):
    return f"<text x='{x}' y='{y}' font-size='{size}' font-weight='{weight}' fill='{fill}' text-anchor='{anchor}' {FONT} {extra}>{esc(s)}</text>"


def multiline(x, y, lines, size=12, fill="#374151", anchor="start", lh=17):
    return "".join(text(x, y + i * lh, ln, size=size, fill=fill, anchor=anchor) for i, ln in enumerate(lines))


def pill(x, y, label, color=None):
    color = color or TOOL_COLORS.get(label, "#374151")
    w = 8 * len(label) + 18
    return (f"<rect x='{x}' y='{y}' rx='11' ry='11' width='{w}' height='22' fill='{color}'/>"
            + text(x + w / 2, y + 15, label, size=11, weight="bold", fill="#fff" if label != "ruff" else "#111", anchor="middle")), w


def pills_row(x, y, labels, max_w):
    out, cx, cy = "", x, y
    for lab in labels:
        p, w = pill(cx, cy, lab)
        if cx + w > x + max_w:
            cx, cy = x, cy + 28
            p, w = pill(cx, cy, lab)
        out += p
        cx += w + 8
    return out, cy + 22


def block(x, y, w, h, title, tools, lines, color="violet", step=None, optional=False):
    bg, accent = COL[color]
    dash = " stroke-dasharray='6 4'" if optional else ""
    s = f"<rect x='{x}' y='{y}' rx='14' ry='14' width='{w}' height='{h}' fill='{bg}' stroke='{accent}' stroke-width='1.5'{dash}/>"
    if step is not None:
        s += f"<circle cx='{x + 20}' cy='{y + 22}' r='13' fill='{accent}'/>" + text(x + 20, y + 27, str(step), size=13, weight="bold", fill="#fff", anchor="middle")
    s += text(x + w / 2, y + 28, title, size=15, weight="bold", fill="#111827", anchor="middle")
    if optional:
        s += text(x + w / 2, y + 44, "optionnel", size=11, fill="#6B7280", anchor="middle")
    p, ny = pills_row(x + 14, y + (54 if optional else 42), tools, w - 28)
    s += p
    s += multiline(x + 14, ny + 20, lines, size=11.5)
    return s


def arrow(x1, y1, x2, y2, label=None, color="#4B5563", dashed=False, label_dy=-6, curve=None):
    dash = " stroke-dasharray='6 4'" if dashed else ""
    if curve:
        path = f"<path d='M{x1},{y1} Q{curve[0]},{curve[1]} {x2},{y2}' fill='none' stroke='{color}' stroke-width='2' marker-end='url(#arr)'{dash}/>"
        lx, ly = curve[0], curve[1]
    else:
        path = f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='2' marker-end='url(#arr)'{dash}/>"
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2
    s = path
    if label:
        for i, ln in enumerate(label.split("\n")):
            s += text(lx, ly + label_dy + i * 14, ln, size=11, fill="#374151", anchor="middle", extra="font-style='italic'")
    return s


def svg(w, h, body, title, subtitle=None):
    head = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>"
            "<defs><marker id='arr' markerWidth='10' markerHeight='10' refX='9' refY='5' orient='auto' markerUnits='strokeWidth'>"
            "<path d='M0,0 L10,5 L0,10 z' fill='#4B5563'/></marker></defs>"
            f"<rect width='{w}' height='{h}' fill='#FFFFFF'/>"
            + text(w / 2, 40, title, size=26, weight="bold", fill="#111827", anchor="middle"))
    if subtitle:
        head += text(w / 2, 64, subtitle, size=13, fill="#6B7280", anchor="middle")
    return head + body + "</svg>"


# ---------------------------------------------------------------- 1. Architecture globale
def diagram_architecture():
    W, H = 1700, 760
    b = ""
    # bandeau CI/CD
    b += f"<rect x='40' y='90' rx='14' ry='14' width='{W - 80}' height='96' fill='#F3F4F6' stroke='#9CA3AF' stroke-width='1.5'/>"
    b += text(60, 116, "CI / CD — GitHub Actions", size=15, weight="bold")
    p, _ = pills_row(60, 128, ["GitHub Actions", "pytest", "ruff", "Docker", "GHCR", "DVC"], 600)
    b += p
    steps = ["git push main", "test : ruff + 36 tests", "validate-model : F1 ≥ 0.75", "build : 4 images → GHCR", "deploy : EC2 AWS via SSM"]
    x = 560
    for i, s_ in enumerate(steps):
        wbox = int(7.5 * len(s_)) + 16
        b += f"<rect x='{x}' y='118' rx='8' ry='8' width='{wbox}' height='34' fill='#fff' stroke='#9CA3AF'/>"
        b += text(x + wbox / 2, 140, s_, size=12, anchor="middle")
        if i < len(steps) - 1:
            b += arrow(x + wbox, 135, x + wbox + 26, 135)
        x += wbox + 30
    b += text(W - 60, 176, "tests ✗ ou F1 < 0.75 → pipeline stoppé, aucune image publiée", size=11, fill="#6B7280", anchor="end")

    # 6 blocs
    y, h, w, gap = 230, 300, 250, 34
    xs = [40 + i * (w + gap) for i in range(6)]
    b += block(xs[0], y, w, h, "1 · Collect & Version", ["Kaggle CSV", "DVC", "DagsHub"],
               ["440 832 lignes (train) + 64 374 (test)", "Preprocessing → 15 features", "dvc.yaml : preprocess → train", "Données + modèle versionnés,", "hash MD5 reproductible"], "violet")
    b += block(xs[1], y, w, h, "2 · Train & Track", ["scikit-learn", "XGBoost", "MLflow"],
               ["3 modèles comparés + tuning", "(RandomizedSearchCV)", "Seuil optimisé sur validation,", "métriques sur hold-out", "Registry churnguard-model", "→ stage Production"], "green")
    b += block(xs[2], y, w, h, "3 · Serve", ["FastAPI", "Pydantic", "Docker", "Prometheus"],
               ["POST /predict · /predict/batch", "GET /model/info · POST /model/reload", "Validation stricte (422)", "Latence loguée + /metrics", "Prédictions → PostgreSQL"], "blue")
    b += block(xs[3], y, w, h, "4 · Orchestrate", ["Airflow"],
               ["batch_scoring — 02:00 quotidien", "auto_retraining — lundi 03:00 :", "check_drift → retrain → evaluate", "→ promote (+ reload API,", "smoke test, rollback si échec)"], "yellow")
    b += block(xs[4], y, w, h, "5 · Monitor & Alert", ["Evidently", "Discord/Slack"],
               ["Référence (train) vs production", "(prédictions reçues / incoming)", "Alerte si > 20 % features en dérive", "ou F1 −0.05", "Rapport JSON + HTML horodaté"], "orange")
    b += block(xs[5], y, w, h, "6 · Store & Leverage", ["PostgreSQL", "Streamlit", "HF Spaces", "Mautic"],
               ["Table predictions (features,", "score, version, latence)", "Dashboard Streamlit (:8501)", "Segments CRM risque élevé /", "moyen (Mautic — simulé)"], "cyan", optional=True)

    # flèches horizontales
    labels = ["dvc pull\ntrain", "modèle\nProduction", "appelle\n/predict/batch", "rapport\nEvidently", "scores +\nsegments"]
    for i in range(5):
        b += arrow(xs[i] + w, y + h / 2, xs[i + 1], y + h / 2, labels[i], label_dy=-12)
    # CI/CD → serve
    b += arrow(xs[2] + w / 2, 186, xs[2] + w / 2, y, "image churnguard-api", label_dy=-4)
    # boucle retour monitor → orchestrate → train
    b += arrow(xs[4] + w / 2, y + h, xs[3] + w / 2, y + h, "", color="#E8641B", curve=(xs[4] + w / 2 - 140, y + h + 90))
    b += text((xs[3] + xs[4] + w) / 2, y + h + 70, "dérive détectée → réentraînement", size=12, weight="bold", fill="#E8641B", anchor="middle")
    b += arrow(xs[3] + w / 2 - 40, y + h + 2, xs[1] + w / 2, y + h + 2, "", color="#1E9E5A", curve=((xs[1] + xs[3]) / 2 + 60, y + h + 120))
    b += text((xs[1] + xs[3] + w) / 2, y + h + 100, "retrain sur référence + fenêtre labellisée · promotion si F1 hold-out meilleur", size=12, weight="bold", fill="#1E9E5A", anchor="middle")
    # producteur temps réel
    b += text(40, y + h + 150, "Producteur de données : SeoLap (SaaS) — en démo, fichier test Kaggle = nouvelles données de production (dérive naturelle vs train)", size=12, fill="#6B7280")
    return svg(W, H, b, "ChurnGuard — Architecture MLOps end-to-end", "Pipeline de prédiction de churn : de la donnée au CRM, avec surveillance et réentraînement automatiques")


# ---------------------------------------------------------------- 2. Boucle de réentraînement
def diagram_retraining():
    W, H = 1500, 720
    b = ""

    def node(x, y, w, h, title, sub, color, diamond=False):
        bg, acc = COL[color]
        if diamond:
            cx, cy = x + w / 2, y + h / 2
            s = f"<polygon points='{cx},{y} {x + w},{cy} {cx},{y + h} {x},{cy}' fill='{bg}' stroke='{acc}' stroke-width='1.5'/>"
        else:
            s = f"<rect x='{x}' y='{y}' rx='12' ry='12' width='{w}' height='{h}' fill='{bg}' stroke='{acc}' stroke-width='1.5'/>"
        s += text(x + w / 2, y + h / 2 - (8 if sub else -5), title, size=14, weight="bold", anchor="middle")
        if sub:
            s += multiline(x + w / 2, y + h / 2 + 10, sub, size=11, anchor="middle", lh=14)
        return s

    # rangée principale
    b += node(60, 120, 200, 90, "Lundi 03:00", ["schedule 0 3 * * 1", "ou trigger manuel"], "grey")
    b += node(320, 100, 240, 130, "check_drift", ["Evidently : référence vs production", "+ nouvelles lignes depuis", "le dernier entraînement"], "orange", diamond=True)
    b += node(640, 120, 220, 90, "retrain_model", ["train(auto_promote=False)", "référence + fenêtre labellisée"], "green")
    b += node(920, 120, 220, 90, "evaluate_model", ["F1 hold-out : candidat", "(son seuil) vs Production"], "green")
    b += node(1200, 100, 240, 130, "decide", ["F1 candidat >", "F1 Production ?"], "yellow", diamond=True)
    b += arrow(260, 165, 320, 165)
    b += arrow(560, 165, 640, 165, "dérive ou\n≥ 5 000 lignes", label_dy=-14)
    b += arrow(860, 165, 920, 165, "best_run_id", label_dy=-8)
    b += arrow(1140, 165, 1200, 165, "new_f1, prod_f1", label_dy=-8)
    # no_retrain
    b += node(340, 300, 200, 70, "no_retrain", ["log · fin du run"], "grey")
    b += arrow(440, 230, 440, 300, "non", label_dy=-4)
    # keep_current
    b += node(1220, 300, 200, 70, "keep_current", ["alerte info · Production inchangée"], "grey")
    b += arrow(1320, 230, 1320, 300, "non", label_dy=-4)
    # promote
    b += node(900, 420, 260, 100, "promote_model", ["register + transition Production", "(ancienne version archivée)"], "green")
    b += arrow(1320, 230, 1160, 470, "oui", curve=(1320, 470), label_dy=-8)
    b += node(560, 420, 260, 100, "POST /model/reload", ["l'API recharge la version", "Production sans redémarrer"], "blue")
    b += arrow(900, 470, 820, 470)
    b += node(220, 420, 260, 100, "smoke test /predict", ["version attendue servie ?", "score ∈ [0, 1] ?"], "blue")
    b += arrow(560, 470, 480, 470)
    b += node(60, 570, 300, 100, "rollback_to_version(prev)", ["ancienne version → Production", "reload API · alerte critique"], "red")
    b += arrow(300, 520, 260, 570, "échec", label_dy=-4)
    b += node(460, 570, 300, 100, "alerte succès", ["Discord/Slack : v4 → v5", "F1 hold-out 0.66 → 0.9x"], "cyan")
    b += arrow(400, 520, 560, 570, "OK", label_dy=-4)
    # légende XCom
    b += text(900, 600, "XCom (scalaires uniquement, jamais de DataFrame) :", size=12, weight="bold")
    b += multiline(900, 620, ["prod_version_before · drift (dict) · new_rows · best_run_id · new_f1 · prod_f1 · new_version"], size=11.5)
    b += text(900, 665, "Fichiers : reports/drift_report.json (+ history/) · données via CHURNGUARD_ROOT monté dans le conteneur", size=11.5)
    return svg(W, H, b, "DAG auto_retraining — de la dérive à la promotion (ou au rollback)", "Airflow · MLflow Registry · FastAPI · Evidently")


# ---------------------------------------------------------------- 3. CI/CD
def diagram_cicd():
    W, H = 1500, 560
    b = ""
    # ligne 1 : code
    b += text(60, 110, "Mise à jour du CODE", size=15, weight="bold", fill="#2F6FE4")
    stages = [
        ("git push main / PR", ["GitHub"], "grey"),
        ("test", ["ruff", "pytest"], "blue"),
        ("validate-model", ["DVC", "joblib"], "green"),
        ("build ×4", ["Docker", "GHCR"], "violet"),
        ("deploy", ["AWS", "SSM"], "yellow"),
    ]
    subs = [
        ["push sur main", "ou pull request"],
        ["lint src/ + tests/", "36 tests : API, preprocessing,", "monitoring, training"],
        ["dvc pull model_artifacts", "F1 ≥ 0.75 sur 500 lignes", "hold-out labellisées"],
        ["churnguard-api, -mlflow,", "-airflow, -dashboard", "tag main-<sha>"],
        ["ssm send-command →", "scripts/deploy.sh : git reset,", "compose build, up -d, reload"],
    ]
    x, y, w, h = 60, 130, 230, 150
    for i, ((title, tools, color), sub) in enumerate(zip(stages, subs)):
        b += block(x, y, w, h, title, tools, sub, color)
        if i < len(stages) - 1:
            b += arrow(x + w, y + h / 2, x + w + 50, y + h / 2, "✓", label_dy=-8)
            b += text(x + w + 25, y + h / 2 + 40, "✗ STOP", size=11, weight="bold", fill="#D64545", anchor="middle")
        x += w + 50
    # ligne 2 : modèle
    b += text(60, 340, "Mise à jour du MODÈLE (indépendante du code)", size=15, weight="bold", fill="#1E9E5A")
    stages2 = [
        ("promotion MLflow", ["MLflow"], "green", ["DAG auto_retraining", "ou registry.py promote"]),
        ("export + dvc push", ["DVC", "DagsHub"], "violet", ["export_model.py →", "model_artifacts/ versionné"]),
        ("tag model-v* / dispatch", ["GitHub Actions"], "grey", ["workflow deploy-model.yml", "validate → bundle"]),
        ("HF Space rebuild", ["HF Spaces", "Docker"], "yellow", ["upload_folder → Space API", "MODEL_PATH standalone"]),
    ]
    x, y = 60, 360
    for i, (title, tools, color, sub) in enumerate(stages2):
        b += block(x, y, w, h, title, tools, sub, color)
        if i < len(stages2) - 1:
            b += arrow(x + w, y + h / 2, x + w + 50, y + h / 2)
        x += w + 50
    b += multiline(1200, 400, ["Secrets GitHub :", "DAGSHUB_USER / DAGSHUB_TOKEN", "HF_TOKEN · AWS_* (SSM, EC2_INSTANCE_ID)", "", "Sans secrets : validation et", "déploiement ignorés avec un", "warning, jamais un faux vert."], size=11.5)
    return svg(W, H, b, "Chaîne CI/CD — deux déclencheurs, deux pipelines", "GitHub Actions · GHCR · DVC · AWS SSM · HuggingFace Spaces")


# ---------------------------------------------------------------- 4. Versioning & lineage
def diagram_versioning():
    W, H = 1500, 600
    b = ""
    b += text(60, 110, "DONNÉES — DVC", size=15, weight="bold", fill="#6B4EE6")
    d = [("raw CSV Kaggle", ["Kaggle CSV"], ["training-master.csv (22 MB)", "testing-master.csv (3 MB)"]),
         ("dvc.yaml : preprocess", ["DVC"], ["clean → features (15)", "split incoming / hold-out"]),
         ("processed/*.csv", ["DVC"], ["features_engineered.csv", "features_incoming.csv", "features_engineered_test.csv"]),
         ("dvc push", ["DagsHub"], ["remote dagshub", "dvc.lock = hash MD5 de", "chaque étape"])]
    x, y, w, h = 60, 130, 250, 140
    for i, (t, tools, sub) in enumerate(d):
        b += block(x, y, w, h, t, tools, sub, "violet")
        if i < len(d) - 1:
            b += arrow(x + w, y + h / 2, x + w + 40, y + h / 2)
        x += w + 40
    b += text(60, 330, "MODÈLE — MLflow", size=15, weight="bold", fill="#1E9E5A")
    m = [("run MLflow", ["MLflow"], ["params · métriques hold-out", "signature · figures · importance", "tags : git_sha, data_dvc_md5"]),
         ("Registry churnguard-model", ["MLflow"], ["v1 … v5", "1 seule version en Production", "les autres archivées"]),
         ("API charge Production", ["FastAPI"], ["models:/churnguard-model/Production", "POST /model/reload à chaud"]),
         ("rollback", ["MLflow"], ["registry.py rollback --version N", "ou automatique (smoke test KO)", "→ reload API < 1 min"])]
    x, y = 60, 350
    for i, (t, tools, sub) in enumerate(m):
        b += block(x, y, w, h, t, tools, sub, "green" if i < 3 else "red")
        if i < len(m) - 1:
            b += arrow(x + w, y + h / 2, x + w + 40, y + h / 2)
        x += w + 40
    # lineage
    b += arrow(60 + 2 * 290 + w / 2, 270, 60 + w / 2, 350, "", color="#6B4EE6", dashed=True, curve=(60 + 290 + 60, 320))
    b += text(60 + 290 + 60, 312, "lineage : tag data_dvc_md5 sur chaque run", size=12, weight="bold", fill="#6B4EE6", anchor="middle")
    b += text(1240, 540, "model_artifacts/ (joblib + model_info.json) est aussi tracké par DVC → CI et HF Space sans serveur MLflow", size=11.5, fill="#6B7280", anchor="end")
    return svg(W, H, b, "Versioning & rollback — données (DVC) et modèles (MLflow)", "Chaque modèle sait sur quelles données il a été entraîné ; chaque version peut revenir en Production")


# ---------------------------------------------------------------- 0. Business
def diagram_business():
    W, H = 1300, 330
    b = ""
    items = [("SeoLap", ["SeoLap"], ["SaaS de monitoring SEO", "utilisateurs en beta"], "grey"),
             ("Signaux d'usage", ["Python"], ["support, retards de paiement,", "usage, ancienneté, dépenses"], "violet"),
             ("ChurnGuard", ["XGBoost", "FastAPI"], ["score de risque par compte", "chaque nuit"], "green"),
             ("CRM", ["Mautic"], ["segments risque élevé / moyen", "(simulé en démo)"], "cyan"),
             ("Action de rétention", ["SeoLap"], ["email, appel, offre —", "avant le départ"], "yellow")]
    x, y, w, h = 40, 100, 220, 150
    for i, (t, tools, sub, c) in enumerate(items):
        b += block(x, y, w, h, t, tools, sub, c)
        if i < len(items) - 1:
            b += arrow(x + w, y + h / 2, x + w + 36, y + h / 2)
        x += w + 36
    b += text(W / 2, 300, "Chaque départ non anticipé = coût d'acquisition perdu. ChurnGuard le signale avant qu'il n'arrive.", size=13, fill="#374151", anchor="middle")
    return svg(W, H, b, "Pourquoi ChurnGuard ?", "Prédire le churn pour agir avant le départ")


# ---------------------------------------------------------------- 5. Production AWS
def diagram_aws():
    W, H = 1500, 730
    b = ""
    # opérateur + GitHub + alertes, en haut
    b += block(40, 90, 330, 118, "Poste opérateur / jury", ["Terraform", "AWS"],
               ["terraform apply → 24 ressources", "SG ouvert : SSH (clé), API, dashboard, UIs"], "grey")
    b += block(430, 90, 400, 118, "GitHub Actions — CI/CD", ["GitHub Actions", "pytest", "Docker", "SSM"],
               ["test → validate-model → build → deploy", "deploy : SSM Run Command → scripts/deploy.sh"], "grey")
    b += block(890, 90, 330, 118, "Alertes", ["Discord/Slack"],
               ["dérive détectée · modèle promu", "échec du DAG"], "orange")

    # cadre AWS
    b += f"<rect x='40' y='250' rx='16' ry='16' width='{W - 80}' height='440' fill='#FFF7EC' stroke='#FF9900' stroke-width='2'/>"
    b += text(60, 278, "AWS eu-north-1 (Stockholm) — VPC par défaut", size=15, weight="bold", fill="#B86E00")
    # cadre EC2
    b += "<rect x='60' y='296' rx='14' ry='14' width='1040' height='376' fill='#FFFFFF' stroke='#FF9900' stroke-width='1.5' stroke-dasharray='6 4'/>"
    b += text(80, 318, "EC2 m7i-flex.large · Ubuntu 24.04 · EBS 30 Go chiffré · rôle d'instance (S3 + SSM) · docker compose prod", size=13, weight="bold")
    y, h, w, gap = 366, 150, 235, 22
    xs = [80 + i * (w + gap) for i in range(4)]
    b += block(xs[0], y, w, h, "API + Dashboard", ["FastAPI", "Streamlit"], ["/predict · /predict/batch · /model/info", "/model/reload · :8000", "Dashboard CRM :8501 (contacts SeoLap)"], "blue")
    b += block(xs[1], y, w, h, "Base de données", ["PostgreSQL"], ["churnguard : predictions", "mlflow : backend store", "airflow : metadata"], "cyan")
    b += block(xs[2], y, w, h, "Orchestration", ["Airflow"], ["batch_scoring (quotidien)", "auto_retraining (dérive OU", "5 000 nouvelles lignes)"], "yellow")
    b += block(xs[3], y, w, h, "Tracking & Registry", ["MLflow"], ["experiment churnguard", "churnguard-model v1 → v2", "artefacts → S3"], "green")
    b += block(80, 532, 480, 126, "Monitoring", ["Evidently", "Prometheus"],
               ["check_drift : référence vs predictions (PostgreSQL)", "/metrics : latence, prédictions par risque, version"], "orange")
    b += block(600, 532, 480, 126, "Bootstrap (user_data.sh, 1er boot)", ["Docker", "Python"],
               ["Docker → clone GitHub → .env (secrets Terraform) → s3 sync", "→ compose build/up → train baseline v1 → reload API"], "violet")
    # S3 + secrets à droite
    b += block(1130, 320, 300, 150, "S3 — chiffré, versionné", ["S3"],
               ["data/processed/*.csv · demo/users.csv", "(référence, incoming, hold-out, contacts)", "mlflow-artifacts/ (modèles, figures)"], "green")
    b += block(1130, 500, 300, 158, "Secrets & accès", ["Terraform"],
               ["mots de passe générés (random)", "clé SSH générée · aucune clé AWS", "sur l'instance (IMDSv2)", "user IAM github-deploy : SSM seul"], "grey")

    # flèches
    b += arrow(205, 208, 205, 250, "apply", label_dy=-4)
    b += arrow(630, 208, 630, 296, "SSM : git pull + build + up", label_dy=-4)
    b += arrow(xs[0] + w, y + 60, xs[1], y + 60)
    b += text(xs[0] + w + gap / 2, y - 8, "predictions", size=11, fill="#374151", anchor="middle", extra="font-style='italic'")
    b += arrow(xs[1] + w, y + 110, xs[2], y + 110)
    b += text(xs[1] + w + gap / 2, y + h + 16, "features récentes (dérive)", size=11, fill="#374151", anchor="middle", extra="font-style='italic'")
    b += arrow(xs[2] + w, y + 60, xs[3], y + 60)
    b += text(xs[2] + w + gap / 2, y - 8, "train / promote", size=11, fill="#374151", anchor="middle", extra="font-style='italic'")
    b += arrow(xs[3] + w, y + 75, 1130, y + 29)
    b += text(1112, y + 36, "artefacts", size=11, fill="#374151", anchor="middle", extra="font-style='italic'")
    b += arrow(xs[2] + w / 2, y, xs[0] + w / 2, y, "", curve=((xs[0] + xs[2] + w) / 2, y - 40))
    b += text((xs[0] + xs[2] + w) / 2, y - 22, "/predict/batch · /model/reload", size=11, fill="#374151", anchor="middle", extra="font-style='italic'")
    b += arrow(1280, 470, 1080, 560, "", color="#6B4EE6", dashed=True)
    b += text(1250, 492, "s3 sync (boot)", size=11, fill="#6B4EE6", anchor="middle", extra="font-style='italic'")
    b += arrow(xs[2] + w - 40, y, 1010, 208, "", color="#E8641B", dashed=True, curve=(1010, 330))
    b += text(1062, 284, "webhook", size=11, fill="#E8641B", anchor="middle", extra="font-style='italic'")
    b += text(W - 60, 712, "≈ 2,5 $/jour · terraform destroy après la soutenance · docs/deployment-aws.md", size=11.5, fill="#6B7280", anchor="end")
    return svg(W, H, b, "Production AWS — le pipeline complet dans le cloud", "Infrastructure as Code (Terraform), déploiement continu (GitHub Actions → SSM), données et artefacts dans S3")

for name, fn in {"01-architecture-globale": diagram_architecture, "02-boucle-reentrainement": diagram_retraining,
                 "03-cicd": diagram_cicd, "04-versioning-lineage": diagram_versioning, "00-business-case": diagram_business,
                 "05-production-aws": diagram_aws}.items():
    (OUT / f"{name}.svg").write_text(fn(), encoding="utf-8")
    print("écrit", OUT / f"{name}.svg")
