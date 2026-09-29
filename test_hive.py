"""python test_hive.py  - fails loudly if the benchmark or the hive stops proving its point."""
from hive.world import _safe_eval
from run import run

quiet = lambda *_: None
res = {m: run(m, log=quiet) for m in ("none", "solo", "hive", "poison")}

assert res["none"]["curve"] == [1, 1, 1]
assert res["hive"]["curve"] == [1, 0, 0], res["hive"]["curve"]
assert res["hive"]["curve"][-1] < res["solo"]["curve"][-1]
assert len(res["hive"]["lessons"]) < len(res["solo"]["lessons"])  # hive learns each lesson once, solo re-learns it
dead = [l for l in res["poison"]["lessons"] if l["dead"]]
assert [l["author"] for l in dead] == ["intruder"], dead  # audit kills the poison and nothing else
assert res["poison"]["curve"][-1] == 0
assert _safe_eval("12.5 + 30/100") == 12.8
try:
    _safe_eval("__import__('os')")
    raise AssertionError("calc must reject code")
except ValueError:
    pass
print("ok")
