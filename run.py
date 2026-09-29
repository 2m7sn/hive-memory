"""Hive Memory - agents that get smarter from each other's mistakes.

5 agents x 3 rounds. Each round:  ACT (parallel) -> AUDIT (counterfactual blame) -> REFLECT -> SYNC to hive.
Same tasks run under four conditions so the effect of shared memory is isolated:
  none    no memory at all
  solo    each agent remembers only its own lessons
  hive    every agent reads every lesson
  poison  hive, plus a hostile lesson injected before round 2 - the audit must catch and kill it
"""
import argparse
import html
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hive.agents import POISON, claude_reflect, claude_solve, sim_reflect, sim_solve, valid_task
from hive.memory import Hive
from hive.world import PLAN, Env, make_task

AGENTS = len(PLAN[0])
COLORS = {"none": "#9ca3af", "solo": "#f59e0b", "hive": "#10b981", "poison": "#ef4444"}


def episode(brain, hive, mode, rnd, i, task, client):
    me = f"agent-{i + 1}"
    lessons = [] if mode == "none" else hive(me).recall(task.tags)
    env = Env(task)
    if brain == "sim":
        ans, used = sim_solve(env, {l["rule"] for l in lessons})
        used_ids = [l["id"] for l in lessons if l["rule"] in used]
    else:
        ans = claude_solve(env, [l["text"] for l in lessons], client)
        used_ids = [l["id"] for l in lessons]  # an LLM can't tell us which lessons it leaned on
    return {"agent": me, "round": rnd + 1, "task": f"{task.source}:{task.traps}", "answer": ans,
            "truth": task.answer, "ok": task.check(ans), "lessons": lessons, "used": used_ids, "env": env}


def audit(hive, ep, task):
    """Counterfactual blame (sim only): replay a failed task with each used lesson removed.
    If dropping a lesson turns the failure into a pass, that lesson is poison - kill it."""
    killed = []
    rules = {l["rule"] for l in ep["lessons"]}
    for l in ep["lessons"]:
        if l["id"] in ep["used"] and task.check(sim_solve(Env(task), rules - {l["rule"]})[0]):
            hive.kill(l["id"])  # already-dead is fine: several agents may convict the same lesson
            killed.append(l)
    return killed


def run(mode, brain="sim", client=None, log=print):
    # solo: one private hive per agent; everything else: one shared hive
    hives = {}
    hive = lambda who: hives.setdefault(who if mode == "solo" else "shared", Hive())
    curve, events, history = [], [], []
    tasks = [[make_task(r, i, valid_task) for i in range(AGENTS)] for r in range(len(PLAN))]
    for rnd in range(len(PLAN)):
        if mode == "poison" and rnd == 1:
            hive("intruder").add(POISON[0], POISON[1], "intruder", 2, rule=POISON[2])
            events.append({"round": 2, "kind": "attack", "agent": "intruder", "text": POISON[0]})
        # ACT - all agents in parallel against the same frozen hive
        with ThreadPoolExecutor(AGENTS) as ex:
            eps = list(ex.map(lambda i: episode(brain, hive, mode, rnd, i, tasks[rnd][i], client), range(AGENTS)))
        # AUDIT + REFLECT + SYNC
        new = 0
        for i, ep in enumerate(eps):
            hive(ep["agent"]).credit(ep["used"], ep["ok"])
            if ep["ok"] or mode == "none":
                continue
            if brain == "sim":
                for l in audit(hive(ep["agent"]), ep, tasks[rnd][i]):
                    if any(e["kind"] == "killed" and e["text"] == l["text"] for e in events):
                        continue
                    events.append({"round": rnd + 1, "kind": "killed", "agent": ep["agent"], "text": l["text"]})
                found = sim_reflect(ep["env"], {l["rule"] for l in ep["lessons"]})
            else:
                found = claude_reflect(ep["env"], ep["answer"], [l["text"] for l in ep["lessons"]], client)
            for rule, text, tags in found:
                _, is_new = hive(ep["agent"]).add(text, tags, ep["agent"], rnd + 1, rule=rule)
                new += is_new
                events.append({"round": rnd + 1, "kind": "learned" if is_new else "confirmed",
                               "agent": ep["agent"], "text": text})
        err = sum(not e["ok"] for e in eps) / AGENTS
        curve.append(err)
        marks = "".join("\033[32m✓\033[0m" if e["ok"] else "\033[31m✗\033[0m" for e in eps)
        log(f"  {mode:<7} round {rnd + 1}  {marks}  error {err:>4.0%}  +{new} lessons")
        for e in eps:
            e.pop("env")
        history += eps
    return {"mode": mode, "curve": curve, "episodes": history, "events": events, "lessons": [l for h in hives.values() for l in h.all()]}


# ---------------------------------------------------------------- report
def chart_svg(results):
    W, H, L, T, R, B = 640, 300, 56, 24, 120, 44
    n = len(PLAN)
    x = lambda r: L + r * (W - L - R) / (n - 1)
    y = lambda e: T + (1 - e) * (H - T - B)
    g = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" '
         f'font-size="12"><rect width="{W}" height="{H}" rx="10" fill="#0f172a"/>']
    for p in (0, .25, .5, .75, 1):
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{y(1 - p)}" y2="{y(1 - p)}" stroke="#334155"/>'
                 f'<text x="{L - 8}" y="{y(1 - p) + 4}" text-anchor="end" fill="#94a3b8">{1 - p:.0%}</text>')
    for r in range(n):
        g.append(f'<text x="{x(r)}" y="{H - 18}" text-anchor="middle" fill="#94a3b8">round {r + 1}</text>')
    g.append(f'<text x="{L}" y="15" fill="#e2e8f0" font-weight="600">error rate (lower is smarter)</text>')
    for k, res in enumerate(results):
        c = COLORS[res["mode"]]
        pts = " ".join(f"{x(r)},{y(e) + (k - 1.5) * 2}" for r, e in enumerate(res["curve"]))
        g.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="3" stroke-linejoin="round"/>')
        g += [f'<circle cx="{x(r)}" cy="{y(e) + (k - 1.5) * 2}" r="4" fill="{c}"/>' for r, e in enumerate(res["curve"])]
        g.append(f'<text x="{W - R + 14}" y="{T + 20 + k * 22}" fill="{c}" font-weight="600">● {res["mode"]}</text>')
    return "\n".join(g + ["</svg>"])


def report_html(results, svg, brain):
    rows = []
    for res in results:
        for l in res["lessons"]:
            status = "☠ killed" if l["dead"] else "alive"
            rows.append(f"<tr class='{'dead' if l['dead'] else ''}'><td>{res['mode']}</td><td>{html.escape(l['text'])}</td>"
                        f"<td>{l['author']}</td><td>{l['born']}</td><td>{l['votes']}</td><td>{l['wins']}/{l['losses']}</td>"
                        f"<td>{status}</td></tr>")
    ev = "".join(f"<li><b>r{e['round']} {e['kind']}</b> · {e['agent']} · {html.escape(e['text'])}</li>"
                 for res in results if res["mode"] == "poison" for e in res["events"])
    summary = " · ".join(f"<b style='color:{COLORS[r['mode']]}'>{r['mode']}</b> {r['curve'][0]:.0%} → {r['curve'][-1]:.0%}"
                         for r in results)
    return f"""<!doctype html><meta charset=utf-8><title>Hive Memory report</title>
<style>body{{background:#020617;color:#e2e8f0;font:15px system-ui;max-width:980px;margin:32px auto;padding:0 16px}}
h1{{margin:0}}.sub{{color:#94a3b8}}table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{border-bottom:1px solid #1e293b;padding:6px;text-align:left}}tr.dead{{color:#f87171;text-decoration:line-through}}
li{{margin:4px 0}}</style>
<h1>🐝 Hive Memory</h1><p class=sub>brain: {brain} · {AGENTS} agents × {len(PLAN)} rounds · {summary}</p>
{svg}<h2>Poison run: what the hive saw</h2><ul>{ev}</ul>
<h2>Every lesson ever written</h2><table><tr><th>run</th><th>lesson</th><th>author</th><th>round</th><th>votes</th><th>wins/losses</th><th>status</th></tr>
{''.join(rows)}</table>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claude", action="store_true", help="real Claude agents (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--modes", default="none,solo,hive,poison")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    modes = a.modes.split(",")
    if bad := set(modes) - set(COLORS):
        ap.error(f"unknown mode(s): {', '.join(bad)} (choose from {', '.join(COLORS)})")
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252 and choke on ✓/🐝
    os.system("")  # switches on ANSI colours in old Windows consoles; no-op elsewhere
    client = None
    if a.claude:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            ap.error("--claude needs ANTHROPIC_API_KEY set (or run without --claude for the free sim)")
        import anthropic
        client = anthropic.Anthropic()
    brain = "claude" if a.claude else "sim"
    print(f"\n🐝 Hive Memory · brain={brain}\n")
    results = [run(m, brain, client) for m in modes]
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    svg = chart_svg(results)
    (out / "chart.svg").write_text(svg, encoding="utf-8")
    (out / "report.html").write_text(report_html(results, svg, brain), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print()
    for r in results:
        print(f"  {r['mode']:<7} error {r['curve'][0]:.0%} → {r['curve'][-1]:.0%}   lessons written: {len(r['lessons'])}")
    print(f"\n  report → {out / 'report.html'}\n")


if __name__ == "__main__":
    main()
