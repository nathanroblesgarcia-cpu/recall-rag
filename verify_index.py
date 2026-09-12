import sqlite3
import config

c = sqlite3.connect(config.DB_PATH)
total = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
notes = c.execute("SELECT COUNT(DISTINCT source) FROM chunks").fetchone()[0]
print(f"total chunks: {total}")
print(f"distinct notes: {notes}\n")

# how many notes per top-level source group
rows = c.execute("SELECT source FROM chunks").fetchall()
groups = {}
for (s,) in rows:
    top = s.split("/", 1)[0]
    groups.setdefault(top, set()).add(s)
for top in sorted(groups):
    print(f"  {top}: {len(groups[top])} notes")
