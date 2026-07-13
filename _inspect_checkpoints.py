import sqlite3
c = sqlite3.connect("checkpoints.sqlite")
tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("TABLES:", tables)
rows = c.execute("SELECT thread_id, checkpoint_ns, checkpoint_id FROM checkpoints WHERE thread_id LIKE '%:lg'").fetchall()
print("ROWS:", rows)
