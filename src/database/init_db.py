from src.database.connection import engine
from src.database.models import Base


def init_db():
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    init_db()
    print("Database inizializzato.")