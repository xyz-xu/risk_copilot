from db.sqlite.auth_dao_impl import *
from db.sqlite.memory_dao_impl import *
from db.sqlite.knowledge_dao_impl import *
from db.sqlite.client_dao_impl import *
from db.sqlite.db_checkpoint_saver import get_sqlite_checkpoint_saver

auth_dao = AuthDaoImpl()
memory_dao = MemoryDaoImpl()
knowledge_dao = KnowledgeDaoImpl()
client_dao = ClientDaoImpl()

checkpointer = get_sqlite_checkpoint_saver()
