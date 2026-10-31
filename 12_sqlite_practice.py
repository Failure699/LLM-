import sqlite3

db = sqlite3.connect("sqlite_practice.sqlite")

db.execute(
    """
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        major TEXT NOT NULL
    )
    """
)
db.execute(
    """
    UPDATE students
    SET major = ?
    WHERE id = ?
    """,
    ("AI应用开发", 2)
)
db.execute(
    """
    DELETE FROM students
    WHERE id = ?
    """,
    (1,)
)
db.commit()

cursor = db.execute(
    """
    SELECT id, name, major
    FROM students
    ORDER BY id
    """
)

rows = cursor.fetchall()

for student_id, name, major in rows:
    print(f"id: {student_id}, name: {name}, major: {major}")

db.close()
