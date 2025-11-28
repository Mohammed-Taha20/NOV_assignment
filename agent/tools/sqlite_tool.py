import sqlite3
import os
from typing import Any , Dict , List
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__) , '..' ,'..', 'data' , 'northwind.sqlite'))

def connect_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def run_sql(query: str) -> List[Dict[str, Any]]:
    query_clean = query.strip()
    # Basic safety check
    if not query_clean.upper().startswith("SELECT"):
        raise ValueError(f"Security: Only SELECT queries allowed. Got: {query_clean[:10]}...")

    conn = connect_db()
    try:
        cur = conn.cursor()
        cur.execute(query_clean)
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        # Re-raise so the Repair node catches it
        raise e
    finally:
        conn.close()
        

def get_schema() -> str:
    """Returns the CREATE TABLE statements."""
    conn = connect_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%';")
        rows = cur.fetchall()
        schema_lines = [row[0] for row in rows if row[0] is not None]
        return "\n\n".join(schema_lines)
    finally:
        conn.close()
