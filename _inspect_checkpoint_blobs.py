import sqlite3
c = sqlite3.connect("checkpoints.sqlite")
cols = c.execute("PRAGMA table_info(checkpoints)").fetchall()
print("COLUMNS:", cols)
rows = c.execute("SELECT thread_id, checkpoint_id, checkpoint, metadata FROM checkpoints WHERE thread_id = '3VERSES_1919_batch_00:ramal:lg'").fetchall()
for r in rows:
    print("thread:", r[0], "checkpoint_id:", r[1])
    print("  checkpoint blob len:", len(r[2]) if r[2] is not None else None, "repr(first 80 bytes):", r[2][:80] if r[2] else r[2])
    print("  metadata blob len:", len(r[3]) if r[3] is not None else None, "repr(first 80 bytes):", r[3][:80] if r[3] else r[3])
