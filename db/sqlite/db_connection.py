import sqlite3
from contextlib import closing

def get_conn():
    with sqlite3.connect("db/sqlite/risk_copilot.db", check_same_thread=False) as conn:
        return conn
