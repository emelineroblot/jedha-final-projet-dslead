"""
Intégrité des DAGs Airflow : import sans erreur, identifiants et dépendances attendus.
Nécessite apache-airflow (absent du job CI `test`, exécuté en local ou dans l'image Airflow).
"""
import sys
from pathlib import Path

import pytest

airflow = pytest.importorskip("airflow")
if sys.platform == "win32":
    pytest.skip("DagBag Airflow 3 dépend de fcntl (POSIX) — exécuter dans le conteneur Airflow", allow_module_level=True)

DAGS_DIR = Path(__file__).parents[1] / "src" / "retraining" / "dags"


@pytest.fixture(scope="module")
def dagbag():
    from airflow.models import DagBag

    return DagBag(dag_folder=str(DAGS_DIR), include_examples=False)


def test_dags_import_without_errors(dagbag):
    assert dagbag.import_errors == {}, dagbag.import_errors
    assert set(dagbag.dag_ids) >= {"batch_scoring", "auto_retraining"}


def test_batch_scoring_structure(dagbag):
    dag = dagbag.get_dag("batch_scoring")
    assert [t.task_id for t in dag.topological_sort()] == ["load_active_accounts", "score_accounts", "push_to_mautic"]


def test_auto_retraining_structure(dagbag):
    dag = dagbag.get_dag("auto_retraining")
    ids = {t.task_id for t in dag.tasks}
    assert ids == {"check_drift", "retrain_model", "evaluate_model", "decide", "promote_model", "keep_current", "no_retrain"}
    assert {t.task_id for t in dag.get_task("check_drift").downstream_list} == {"retrain_model", "no_retrain"}
    assert {t.task_id for t in dag.get_task("decide").downstream_list} == {"promote_model", "keep_current"}
    assert dag.get_task("retrain_model").downstream_task_ids == {"evaluate_model"}


def test_schedules(dagbag):
    assert str(dagbag.get_dag("batch_scoring").schedule_interval or dagbag.get_dag("batch_scoring").timetable.summary) in ("0 2 * * *",)
    assert str(dagbag.get_dag("auto_retraining").schedule_interval or dagbag.get_dag("auto_retraining").timetable.summary) in ("0 3 * * 1",)
