import sqlite3
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class AuditReport:
    missing: List[str] = field(default_factory=list)
    orphans: List[Path] = field(default_factory=list)
    corrupted: List[Tuple[str, str]] = field(default_factory=list)  # (doc_id, path)
    healthy_count: int = 0


class LibrarianAuditor:
    def __init__(self, agent):
        self.agent = agent
        self.report = AuditReport()

    def _calculate_hash(self, path: Path) -> str:
        """Helper to hash files during the scan."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def run_full_audit(self):
        """Reconciles the database with the filesystem."""
        self.report = AuditReport()
        db_files = {}  # {rel_path: (doc_id, expected_sha)}

        # 1. Map the Database State
        with sqlite3.connect(self.agent.db_path) as conn:
            cursor = conn.execute("SELECT doc_id, local_path, sha256 FROM documents")
            for row in cursor:
                db_files[row[1]] = (row[0].hex(), row[2])

        # 2. Scan the Physical Filesystem
        # We look for all files inside the /data directory
        found_on_disk = set()
        for file_path in self.agent.storage_path.rglob("*"):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(self.agent.root))
                found_on_disk.add(rel_path)

                if rel_path in db_files:
                    doc_id_hex, expected_sha = db_files[rel_path]
                    # Verify Content Integrity
                    actual_sha = self._calculate_hash(file_path)
                    if actual_sha == expected_sha:
                        self.report.healthy_count += 1
                    else:
                        self.report.corrupted.append((doc_id_hex, rel_path))
                else:
                    self.report.orphans.append(file_path)

        # 3. Identify Missing Files
        for rel_path in db_files:
            if rel_path not in found_on_disk:
                self.report.missing.append(rel_path)

        return self.report

    def print_summary(self):
        r = self.report
        print(f"--- Library Audit Summary ---")
        print(f"✅ Healthy Files:   {r.healthy_count}")
        print(f"❌ Missing Files:   {len(r.missing)}")
        print(f"⚠️  Corrupted Files: {len(r.corrupted)}")
        print(f"❓ Orphaned Files:  {len(r.orphans)}")

        if r.corrupted:
            print("\nCorrupted (Hash Mismatch):")
            for cid, path in r.corrupted:
                print(f"  - {path} (ID: {cid})")

        if r.missing:
            print("\nMissing from Disk:")
            for path in r.missing:
                print(f"  - {path}")
