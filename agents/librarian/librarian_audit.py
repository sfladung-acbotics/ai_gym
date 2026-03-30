import hashlib
import json
import sqlite3
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict


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
        - If 3+ levels deep (xx/yy/hash/), moves the hash dir.
        - If shallow (data/e5/file.txt), moves ONLY the file.
        """
        try:
            rel_to_storage = file_path.relative_to(self.agent.storage_path)
            # Standard path: xx/yy/hash_string/file.pdf (4 parts)
            if len(rel_to_storage.parts) >= 3:
                return self.agent.storage_path.joinpath(*rel_to_storage.parts[:3])
            return file_path
        except (ValueError, IndexError):
            return file_path

    def _quarantine_entry(self, file_path: Path, reason: str):
        target = self._get_quarantine_target(file_path)
        if target == self.agent.storage_path or target == self.quarantine_path:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{reason}_{timestamp}_{target.name}"
        dest = self.quarantine_path / folder_name

        try:
            shutil.move(str(target), str(dest))
            print(f" [!] Quarantined: {target.name} -> {dest.name}")
        except Exception as e:
            print(f" [!] Error moving {target}: {e}")

    def _attempt_redownload(self, doc_id_hex: str):
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
                    print(f" [->] Restoring: {url}")
                    self.agent.add_remote_document(url, title=title, tags=tags)
                except Exception as e:
                    print(f" [!] Restoration failed: {e}")

    def run_full_audit(self, repair: bool = False):
        self.report = AuditReport()
        db_entries = {}
        processed_targets = set()

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
            if target in processed_targets:
                continue

            problem_found, reason = False, ""
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

        # 2. Missing Check
        for rel_path, (doc_id_hex, _) in db_entries.items():
            if not (self.agent.root / rel_path).exists():
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
