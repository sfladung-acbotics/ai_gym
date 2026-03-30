import hashlib
import json
import sqlite3
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class AuditReport:
    healthy_count: int = 0
    missing: List[Dict[str, str]] = field(default_factory=list)
    orphans: List[Path] = field(default_factory=list)
    corrupted: List[Dict[str, str]] = field(default_factory=list)


class LibrarianAuditor:
    def __init__(self, agent):
        self.agent = agent
        self.quarantine_path = self.agent.storage_path / "quarantine"
        self.report = AuditReport()
        self.quarantine_path.mkdir(parents=True, exist_ok=True)

    def _calculate_hash(self, path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_quarantine_target(self, file_path: Path) -> Path:
        """
        Surgically identifies the target to move.
        - If the file is 3+ levels deep (standard hash dir), move the hash dir.
        - If the file is shallow (e.g., /data/e5/file.txt), move ONLY the file.
        """
        try:
            rel_to_storage = file_path.relative_to(self.agent.storage_path)

            # Standard path: xx/yy/hash_string/file.pdf (4 parts)
            # If we have 3 or more parts, we assume it's an Agent-managed directory.
            if len(rel_to_storage.parts) >= 3:
                # We want the directory that is the 'hash' folder
                # e.g., parts is ('e5', 'ab', 'hash_string', 'file.pdf')
                # target is storage / 'e5' / 'ab' / 'hash_string'
                return self.agent.storage_path.joinpath(*rel_to_storage.parts[:3])

            # If it's shallow (e.g., 'e5/file.txt' or 'file.txt'), move only the file.
            return file_path

        except (ValueError, IndexError):
            return file_path

    def _quarantine_entry(self, file_path: Path, reason: str):
        """Moves the top-level entry containing the file to quarantine."""
        target = self._get_quarantine_target(file_path)

        # Guard: Never move the storage root or the quarantine folder itself
        if target == self.agent.storage_path or target == self.quarantine_path:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{reason}_{timestamp}_{target.name}"
        dest = self.quarantine_path / folder_name

        try:
            shutil.move(str(target), str(dest))
            print(f" [!] Quarantined: {target.name} -> {dest.name}")
        except Exception as e:
            print(f" [!] Error moving {target} to quarantine: {e}")

    def _attempt_redownload(self, doc_id_hex: str):
        """Attempts full restoration via the Agent's remote ingest pipeline."""
        with sqlite3.connect(self.agent.db_path) as conn:
            query = """
                SELECT p.source_url, d.title, d.domain_data 
                FROM provenance p
                JOIN documents d ON p.doc_id = d.doc_id
                WHERE d.doc_id = ? 
                AND p.source_url NOT LIKE 'local_filesystem'
                ORDER BY p.retrieved_at DESC LIMIT 1
            """
            row = conn.execute(query, (bytes.fromhex(doc_id_hex),)).fetchone()

            if row:
                url, title, tags_json = row
                tags = json.loads(tags_json) if tags_json else {}
                try:
                    # Re-ingest handles hashing and path recreation
                    print(f" [->] Restoring: {url}")
                    new_id = self.agent.add_remote_document(url, title=title, tags=tags)
                    if new_id == doc_id_hex:
                        print(f" [+] Verified & Restored: {doc_id_hex}")
                    else:
                        print(
                            f" [!] Warning: New download ID {new_id} differs from original."
                        )
                except Exception as e:
                    print(f" [!] Redownload failed: {e}")

    def run_full_audit(self, repair: bool = False):
        self.report = AuditReport()
        db_entries = {}
        processed_targets = set()  # Avoid moving same parent dir multiple times

        with sqlite3.connect(self.agent.db_path) as conn:
            cursor = conn.execute("SELECT doc_id, local_path, sha256 FROM documents")
            for row in cursor:
                db_entries[row[1]] = (row[0].hex(), row[2])

        # 1. Physical Scan
        for file_path in self.agent.storage_path.rglob("*"):
            if not file_path.is_file() or self.quarantine_path in file_path.parents:
                continue

            rel_path = str(file_path.relative_to(self.agent.root))
            target = self._get_quarantine_target(file_path)

            # If we already quarantined this parent directory in this run, skip
            if target in processed_targets:
                continue

            problem_found = False
            reason = ""

            if rel_path in db_entries:
                doc_id_hex, expected_sha = db_entries[rel_path]
                if self._calculate_hash(file_path) == expected_sha:
                    self.report.healthy_count += 1
                else:
                    self.report.corrupted.append(
                        {"rel_path": rel_path, "doc_id_hex": doc_id_hex}
                    )
                    problem_found, reason = True, "corrupt"
            else:
                self.report.orphans.append(file_path)
                problem_found, reason = True, "orphan"

            if problem_found and repair:
                self._quarantine_entry(file_path, reason)
                processed_targets.add(target)

        # 2. Re-verify Missing & Repair
        for rel_path, (doc_id_hex, _) in db_entries.items():
            # Check if it was moved during the physical scan or is just gone
            abs_path = self.agent.root / rel_path
            if not abs_path.exists():
                self.report.missing.append(
                    {"rel_path": rel_path, "doc_id_hex": doc_id_hex}
                )
                if repair:
                    self._attempt_redownload(doc_id_hex)

        return self.report

    def print_summary(self):
        r = self.report
        print(f"\n{'='*40}")
        print(f"LIBRARIAN AUDIT REPORT: {datetime.now().isoformat()}")
        print(f"{'='*40}")
        print(f"✅ Healthy:   {r.healthy_count}")
        print(f"❌ Missing:   {len(r.missing)}")
        print(f"⚠️  Corrupted: {len(r.corrupted)}")
        print(f"❓ Orphans:   {len(r.orphans)}")
        print(f"{'='*40}\n")
