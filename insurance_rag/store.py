"""PostgreSQL + pgvector store: HNSW index for cosine vector search, GIN index over a generated tsvector for keywords."""
from __future__ import annotations

import os
import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

from insurance_rag.chunking import Chunk

DEFAULT_DSN = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/ragtest")
_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,40}$")
_COLS = "id, doc_id, plan, source, page, headings, text"


class ChunkStore:
    def __init__(self, dim: int, dsn: str = DEFAULT_DSN, table: str = "chunks", recreate: bool = False):
        if not _IDENT.match(table):
            raise ValueError(f"bad table name {table!r}")
        self.dim, self.table = dim, table
        self.t = sql.Identifier(table)
        self.conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        if recreate:
            self.conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self.t))
        self._create()

    def _create(self):
        c = self.conn
        c.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {t} (
              id BIGSERIAL PRIMARY KEY, doc_id TEXT NOT NULL, plan TEXT NOT NULL, source TEXT NOT NULL,
              page INT, headings TEXT[] NOT NULL DEFAULT '{{}}', heading_text TEXT NOT NULL DEFAULT '', text TEXT NOT NULL,
              embedding vector({dim}) NOT NULL,
              tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('english'::regconfig, heading_text || ' ' || text)) STORED)
            """).format(t=self.t, dim=sql.Literal(self.dim)))
        c.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING hnsw (embedding vector_cosine_ops)")
                  .format(i=sql.Identifier(self.table + "_hnsw"), t=self.t))
        c.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING gin (tsv)")
                  .format(i=sql.Identifier(self.table + "_gin"), t=self.t))
        c.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} (plan)")
                  .format(i=sql.Identifier(self.table + "_plan"), t=self.t))
        have = c.execute("SELECT atttypmod AS d FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'",
                         (self.table,)).fetchone()["d"]
        if have != self.dim:
            raise ValueError(f"table '{self.table}' stores {have}-dim vectors but the embedder produces {self.dim}; "
                             "re-ingest with recreate=True")

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Insert chunks. Re-ingesting a doc_id replaces its old chunks (idempotent)."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        if len(chunks) and vectors.shape[1] != self.dim:
            raise ValueError(f"vectors are {vectors.shape[1]}-dim, store is {self.dim}")
        with self.conn.transaction():
            for d in {c.doc_id for c in chunks}:
                self.conn.execute(sql.SQL("DELETE FROM {} WHERE doc_id = %s").format(self.t), (d,))
            with self.conn.cursor() as cur:
                cur.executemany(
                    sql.SQL("INSERT INTO {} (doc_id, plan, source, page, headings, heading_text, text, embedding) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)").format(self.t),
                    [(c.doc_id, c.plan, c.source, c.page, c.headings, " ".join(c.headings), c.text, v.astype(np.float32))
                     for c, v in zip(chunks, vectors)])
        return len(chunks)

    def count(self) -> int:
        return self.conn.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(self.t)).fetchone()["n"]

    def plans(self) -> list[str]:
        return [r["plan"] for r in self.conn.execute(
            sql.SQL("SELECT DISTINCT plan FROM {} ORDER BY plan").format(self.t))]

    def vector_search(self, qvec: np.ndarray, k: int = 10, plan: str | None = None, exact: bool = False,
                      ef_search: int = 100) -> list[dict]:
        """Cosine nearest neighbours. exact=True disables the HNSW index (sequential scan) for ground truth."""
        where = sql.SQL("WHERE plan = %s") if plan else sql.SQL("")
        q = sql.SQL("SELECT {cols}, 1 - (embedding <=> %s) AS score FROM {t} {w} ORDER BY embedding <=> %s LIMIT %s").format(
            cols=sql.SQL(_COLS), t=self.t, w=where)
        v = np.asarray(qvec, dtype=np.float32)
        args = [v] + ([plan] if plan else []) + [v, k]
        with self.conn.transaction():
            if exact:
                self.conn.execute("SET LOCAL enable_indexscan = off")
            else:
                self.conn.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(int(ef_search))))
            return self.conn.execute(q, args).fetchall()

    def keyword_search(self, query: str, k: int = 10, plan: str | None = None) -> list[dict]:
        """Full-text search; query words are OR-ed (natural-language questions rarely match every word)."""
        toks = sorted(set(re.findall(r"[a-z0-9]+", query.lower())))
        if not toks:
            return []
        tsq = " | ".join(toks)
        where = sql.SQL(" AND plan = %s") if plan else sql.SQL("")
        q = sql.SQL("SELECT {cols}, ts_rank_cd(tsv, qry) AS score FROM {t}, to_tsquery('english', %s) qry "
                    "WHERE tsv @@ qry{w} ORDER BY score DESC, id LIMIT %s").format(cols=sql.SQL(_COLS), t=self.t, w=where)
        return self.conn.execute(q, [tsq] + ([plan] if plan else []) + [k]).fetchall()

    def close(self):
        self.conn.close()
