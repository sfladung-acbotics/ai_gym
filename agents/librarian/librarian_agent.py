import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


from web_downloader import WebDownloader

# We use a fixed Namespace for UUID v5 to ensure different machines
# generate the same UUID for the same file hash.
LIB_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
CURRENT_SCHEMA_VERSION = 1


class LibrarianAgent:
    def __init__(
        self,
        library_root: str,
        agent_id: str = "default_agent",
        downloader_config: dict = None,
    ):
        self.agent_id = agent_id

        # Instantiate downloader with optional overrides
        config = downloader_config or {}
        self.downloader = WebDownloader(agent_id=agent_id, **config)
        self.agent_id = agent_id  # Identification for this specific instance
        self.root = Path(library_root).expanduser().resolve()
        self.db_path = self.root / "librarian.sqlite"
        self.storage_path = self.root / "data"

        # 1. Physical Initialization
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 2. Logical Initialization (Database & Schema)
        self._init_db()

    def _init_db(self):
        """Initializes SQLite with Pragma versioning and core tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")

            # Check schema version
            cursor = conn.execute("PRAGMA user_version;")
            v = cursor.fetchone()[0]

            if v == 0:
                self._apply_base_schema(conn)
            elif v < CURRENT_SCHEMA_VERSION:
                self._run_migrations(conn, v)

    def _apply_base_schema(self, conn):
        """Applies the normalized, audit-capable schema."""
        # Core Document Table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id BLOB PRIMARY KEY,
                sha256 TEXT UNIQUE NOT NULL,
                metadata_hash TEXT,
                file_size INTEGER,
                title TEXT,
                local_path TEXT,
                added_at TIMESTAMP,
                domain_data TEXT 
            )
        """
        )

        # History Ledger for Merging/Auditing
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_history (
                history_id BLOB PRIMARY KEY,
                doc_id BLOB,
                parent_hash TEXT,
                metadata_hash TEXT,
                snapshot TEXT,
                change_note TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            )
        """
        )

        # Relationship Edges (Knowledge Graph)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_edges (
                source_id BLOB,
                target_id BLOB,
                edge_type TEXT,
                PRIMARY KEY (source_id, target_id, edge_type),
                FOREIGN KEY(source_id) REFERENCES documents(doc_id),
                FOREIGN KEY(target_id) REFERENCES documents(doc_id)
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS provenance (
                prov_id BLOB PRIMARY KEY,
                doc_id BLOB,
                source_url TEXT,
                retriever_id TEXT,
                retrieved_at TIMESTAMP,
                metadata_snapshot TEXT, -- Headers or context at time of retrieval
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
            )
        """
        )
        # Indexing for quick lookups of sources
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prov_doc ON provenance(doc_id);")
        conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION};")
        print(f"Initialized Librarian DB at v{CURRENT_SCHEMA_VERSION}")

    def _calculate_file_identity(self, file_path: Path) -> Tuple[str, int, uuid.UUID]:
        """Calculates SHA256 and the deterministic UUIDv5."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)

        hash_str = sha256_hash.hexdigest()
        doc_uuid = uuid.uuid5(LIB_NAMESPACE, hash_str)
        return hash_str, file_path.stat().st_size, doc_uuid

    def add_remote_document(
        self, url: str, title: Optional[str] = None, tags: dict = None
    ):
        """High-level command to fetch and store a web document."""
        return self.downloader.fetch_to_callback(
            url=url,
            callback=self.ingest,  # The Agent's own ingest method
            title=title,
            tags=tags,
        )

    def ingest(
        self,
        file_path: str,
        source_url: str = "local_filesystem",
        title: Optional[str] = None,
        tags: Dict[str, Any] = None,
        retriever: Optional[str] = None,
    ):
        """
        Ingests a file and records its unique provenance.
        """
        src = Path(file_path)
        sha256, size, doc_id = self._calculate_file_identity(src)
        doc_title = title or src.name
        retriever_id = retriever or self.agent_id

        # 1. Storage logic (as before)
        rel_dir = Path(sha256[:2]) / sha256[2:4] / sha256
        abs_dir = self.storage_path / rel_dir
        abs_dir.mkdir(parents=True, exist_ok=True)
        dest_path = abs_dir / src.name
        if not dest_path.exists():
            shutil.copy2(src, dest_path)

        domain_blob = json.dumps(tags or {}, sort_keys=True)
        meta_hash = hashlib.sha256(domain_blob.encode()).hexdigest()

        with sqlite3.connect(self.db_path) as conn:
            # 2. Update/Insert Document
            conn.execute(
                """
                INSERT INTO documents (doc_id, sha256, metadata_hash, file_size, title, local_path, added_at, domain_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET title=coalesce(title, excluded.title)
            """,
                (
                    doc_id.bytes,
                    sha256,
                    meta_hash,
                    size,
                    doc_title,
                    str(dest_path.relative_to(self.root)),
                    datetime.now(),
                    domain_blob,
                ),
            )

            # 3. Record this specific Provenance event
            # We use a random UUID for the event because one doc can have many sources.
            conn.execute(
                """
                INSERT INTO provenance (prov_id, doc_id, source_url, retriever_id, retrieved_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    uuid.uuid4().bytes,
                    doc_id.bytes,
                    source_url,
                    retriever_id,
                    datetime.now(),
                ),
            )

        return doc_id.hex

    def query_by_tag(self, key: str, value: Any):
        """Simple demonstration of querying the JSONB blob."""
        with sqlite3.connect(self.db_path) as conn:
            # Note: json_extract is used for SQLite's JSON support
            query = "SELECT title, domain_data FROM documents WHERE json_extract(domain_data, ?) = ?"
            cursor = conn.execute(query, (f"$.{key}", value))
            return cursor.fetchall()


if __name__ == "__main__":
    librarian = LibrarianAgent("~/my_tech_library")
    print(librarian)
    librarian.add_remote_document("https://www.ti.com/lit/ds/symlink/ina950-sep.pdf")
