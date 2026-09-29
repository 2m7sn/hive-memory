"""A small, hostile data world. Every task asks the same question (total sales for a month),
but each data source hides different traps that silently give wrong answers."""
import ast
import csv
import io
import operator
import random
from dataclasses import dataclass

# trap letter -> the rule that defeats it
TRAPS = {"R": "retry_429", "P": "paginate", "C": "cents", "D": "dmy", "S": "sniff_delimiter", "K": "decimal_comma"}

# PLAN[round][agent] = (source, traps). Each agent keeps meeting traps it has never seen,
# so only shared memory can prepare it.
PLAN = [
    [("api", "R"), ("api", "PC"), ("api", "D"), ("csv", "S"), ("csv", "K")],
    [("api", "PD"), ("csv", "SK"), ("api", "RC"), ("api", "CD"), ("csv", "SD")],
    [("csv", "SKD"), ("api", "RD"), ("csv", "KS"), ("api", "RP"), ("api", "PCR")],
]
MONTHS = "January February March April May June July August September October November December".split()


@dataclass
class Task:
    source: str
    traps: str
    records: list
    month: int
    answer: float

    @property
    def prompt(self):
        where = ("the orders API (start with http_get('/orders?page=1'))" if self.source == "api"
                 else "the CSV export (read_file('exports/orders.csv'))")
        return (f"What is the total order amount in SAR for {MONTHS[self.month - 1]} 2026, according to {where}? "
                "Submit a number rounded to 2 decimals.")

    @property
    def tags(self):
        return [self.source, "dates", "amounts"]

    def check(self, ans):
        return ans is not None and abs(float(ans) - self.answer) < 0.01


class Env:
    """Tool surface for one episode. Every call is recorded in .trace."""
    PAGE = 4

    def __init__(self, task):
        self.t, self.trace, self.limited = task, [], False

    def _date(self, y, m, d):
        return f"{d:02d}/{m:02d}/{y}" if "D" in self.t.traps else f"{y}-{m:02d}-{d:02d}"

    def http_get(self, path):
        t = self.t
        if t.source != "api" or not path.startswith("/orders"):
            out = {"status": 404, "error": "not found"}
        elif "R" in t.traps and not self.limited:
            self.limited = True
            out = {"status": 429, "error": "Too Many Requests"}
        else:
            page = int(path.split("page=")[1]) if "page=" in path else 1
            size = self.PAGE if "P" in t.traps else len(t.records)
            chunk = t.records[(page - 1) * size: page * size]
            data = [{"id": r[0], "date": self._date(*r[1]),
                     **({"amount_cents": round(r[2] * 100)} if "C" in t.traps else {"amount": r[2]})} for r in chunk]
            more = page * size < len(t.records)
            out = {"status": 200, "data": data, "next": f"/orders?page={page + 1}" if more else None}
        self.trace.append({"tool": "http_get", "arg": path, "result": out})
        return out

    def read_file(self, path):
        t = self.t
        if t.source != "csv" or path != "exports/orders.csv":
            out = "ERROR: file not found"
        else:
            buf = io.StringIO()
            w = csv.writer(buf, delimiter=";" if "S" in t.traps else ",", lineterminator="\n")
            w.writerow(["id", "date", "amount"])
            for id_, ymd, amt in t.records:
                w.writerow([id_, self._date(*ymd), f"{amt:.2f}".replace(".", ",") if "K" in t.traps else f"{amt:.2f}"])
            out = buf.getvalue()
        self.trace.append({"tool": "read_file", "arg": path, "result": out})
        return out

    def calc(self, expr):
        return _safe_eval(expr)

    def call(self, name, args):
        try:
            return {"http_get": lambda: self.http_get(args["path"]), "read_file": lambda: self.read_file(args["path"]),
                    "calc": lambda: self.calc(args["expr"])}[name]()
        except Exception as e:
            return f"ERROR: {e}"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(expr):
    """Arithmetic only. The LLM never gets eval()."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only arithmetic allowed")
    return round(ev(ast.parse(expr, mode="eval").body), 6)


def make_task(rnd, agent, validate):
    """Deterministic task for (round, agent). Re-rolls the data until every trap it carries really bites:
    validate(task) must confirm naive solving fails and each trap's rule is necessary."""
    source, traps = PLAN[rnd][agent]
    for seed in range(rnd * 10 + agent, 10_000, 100):
        rng = random.Random(seed)
        recs = [(i + 1, (2026, rng.randint(1, 12), rng.randint(1, 28)), round(rng.uniform(20, 900), 2))
                for i in range(10)]
        month = recs[rng.randrange(10)][1][1]
        answer = round(sum(a for _, (_, m, _), a in recs if m == month), 2)
        task = Task(source, traps, recs, month, answer)
        if validate(task):
            return task
    raise RuntimeError(f"no valid data for round {rnd} agent {agent}")
