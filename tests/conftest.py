import os
import uuid

import psycopg
import pytest

from insurance_rag import sample_docs
from insurance_rag.embeddings import HashingEmbedder
from insurance_rag.ingest import ingest
from insurance_rag.retrieval import Retriever
from insurance_rag.store import DEFAULT_DSN, ChunkStore

DSN = os.environ.get("TEST_DATABASE_URL", DEFAULT_DSN)


def _db_available() -> bool:
    try:
        psycopg.connect(DSN, connect_timeout=3).close()
        return True
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _db_available(), reason="no PostgreSQL+pgvector at TEST_DATABASE_URL/DATABASE_URL")


@pytest.fixture
def make_store():
    made = []

    def _make(dim: int):
        s = ChunkStore(dim, DSN, table="t_" + uuid.uuid4().hex[:10])
        made.append(s)
        return s

    yield _make
    for s in made:
        s.conn.execute(f"DROP TABLE IF EXISTS {s.table}")
        s.close()


@pytest.fixture(scope="session")
def sample(tmp_path_factory):
    return sample_docs.build(str(tmp_path_factory.mktemp("sample")))


@pytest.fixture
def loaded(make_store, sample):
    """Both fictional plans ingested (pypdf + hashing embedder) into a throwaway table."""
    emb = HashingEmbedder()
    store = make_store(emb.dim)
    ingest([(p, name) for name, p in sample["pdfs"].items()], "pypdf", emb, store)
    return Retriever(store, emb), store, emb
