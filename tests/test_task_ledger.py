from scripts.lib.task_ledger import TaskLedger


def test_task_ledger_resumes_and_updates_active_goal(tmp_path):
    ledger = TaskLedger(tmp_path / "tasks.json")
    first = ledger.start("Fix the tests", resume=True)
    resumed = ledger.start("Here is the next error", resume=True)
    assert resumed["id"] == first["id"]
    assert "next error" in resumed["updates"][-1]["note"]
    completed = ledger.update(first["id"], status="complete", note="All tests pass")
    assert completed["status"] == "complete"
    assert ledger.active_task() is None


def test_task_ledger_new_task_does_not_reuse_active_goal(tmp_path):
    ledger = TaskLedger(tmp_path / "tasks.json")
    first = ledger.start("First", resume=False)
    second = ledger.start("Second", resume=False)
    assert first["id"] != second["id"]
