import sqlite3


DATABASE_URI = "file:checkpoints_10.sqlite?mode=ro"
TARGET_THREAD = "user_002_chat_2"

db = sqlite3.connect(
    DATABASE_URI,
    uri=True
)

cursor = db.execute(
    """
    SELECT thread_id, COUNT(*)
    FROM checkpoints
    GROUP BY thread_id
    ORDER BY thread_id
    """
)
thread_rows = cursor.fetchall()

for thread_id, count in thread_rows:
    print(f"线程：{thread_id} checkpoints数量： {count}")

cursor2 = db.execute(
    """
    SELECT checkpoint_id, parent_checkpoint_id
    FROM checkpoints
    WHERE thread_id = ?
    ORDER BY checkpoint_id DESC
    LIMIT 3
    """,
    (TARGET_THREAD,)
)
checkpoint_rows = cursor2.fetchall()

for checkpoint_id, parent_checkpoint_id in checkpoint_rows:
    print(f"当前checkpoint：{checkpoint_id}")
    print(f"上一个checkpoint：{parent_checkpoint_id}")

db.close()
