"""The hive: one shared lessons store every agent reads before acting and writes after failing."""
import json
import sqlite3
import threading
from difflib import SequenceMatcher


class Hive:
    def __init__(self, path=":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("""create table if not exists lessons(
            id integer primary key, text text, rule text, tags text, author text, born int,
            votes int default 1, wins int default 0, losses int default 0, dead int default 0)""")

    def add(self, text, tags, author, born, rule=None):
        """Insert a lesson, or upvote it if the hive already knows it. Returns (id, is_new)."""
        with self.lock:
            for id_, t, r in self.db.execute("select id, text, rule from lessons where dead=0").fetchall():
                if (rule and rule == r) or SequenceMatcher(None, t.lower(), text.lower()).ratio() > 0.8:
                    self.db.execute("update lessons set votes = votes + 1 where id=?", (id_,))
                    return id_, False
            cur = self.db.execute("insert into lessons(text, rule, tags, author, born) values(?,?,?,?,?)",
                                  (text, rule, json.dumps(tags), author, born))
            return cur.lastrowid, True

    def recall(self, task_tags, k=8):
        """Top-k live lessons relevant to the task."""
        # ponytail: tag overlap + track record, swap for embeddings once lessons are free text at scale
        with self.lock:
            rows = self.db.execute("select id, text, rule, tags, votes, wins, losses, author from lessons "
                                   "where dead=0").fetchall()
        scored = []
        for id_, text, rule, tags, votes, wins, losses, who in rows:
            overlap = len(set(json.loads(tags)) & set(task_tags))
            if overlap:
                scored.append((overlap + 0.2 * (wins - losses) + 0.1 * votes,
                               {"id": id_, "text": text, "rule": rule}))
        return [l for _, l in sorted(scored, key=lambda s: -s[0])[:k]]

    def credit(self, ids, ok):
        with self.lock:
            col = "wins" if ok else "losses"
            self.db.executemany(f"update lessons set {col} = {col} + 1 where id=?", [(i,) for i in ids])

    def kill(self, id_):
        with self.lock:
            self.db.execute("update lessons set dead=1 where id=?", (id_,))

    def all(self):
        cols = ["id", "text", "rule", "tags", "author", "born", "votes", "wins", "losses", "dead"]
        with self.lock:
            return [dict(zip(cols, r)) for r in self.db.execute(f"select {','.join(cols)} from lessons")]
