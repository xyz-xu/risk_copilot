from langgraph.checkpoint.sqlite import SqliteSaver
from db.sqlite.db_connection import get_conn

def get_sqlite_checkpoint_saver():
    return SqliteSaver(get_conn())
