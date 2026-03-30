import time
import requests
import tempfile
from pathlib import Path
from typing import Callable, Any, Optional, Dict
from functools import wraps
from urllib.parse import urlparse, urlunparse
import os

ALLOWED_MIMES = [
    "application/pdf",
    "application/octet-stream",  # Often used for firmware/binary blobs
    "text/plain",
    "application/zip",
]


def retry_download(method):
    """Decorator to handle exponential backoff for downloads."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        last_exception = None
        # Start with the initial delay
        delay = self.retry_delay

        for attempt in range(self.max_retries + 1):
            try:
                return method(self, *args, **kwargs)
            except (requests.exceptions.RequestException, IOError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    print(
                        f"[{self.agent_id}] Attempt {attempt + 1} failed. "
                        f"Retrying in {delay}s... (Error: {e})"
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(
                        f"[{self.agent_id}] All {self.max_retries + 1} attempts failed."
                    )

        raise last_exception

    return wrapper


class WebDownloader:
    def __init__(
        self,
        agent_id: str,
        max_retries: int = 3,
        retry_delay: int = 2,
        timeout: int = 45,
    ):
        self.agent_id = agent_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": f"LibrarianAgent/{agent_id}"})

    @retry_download
    def fetch_to_callback(
        self, url: str, callback: Callable[[Path, str], Any], **kwargs
    ):
        # Normalize the URL for the callback/provenance
        canonical_url = self.normalize_url(url)

        with tempfile.TemporaryDirectory() as tmp_dir:
            parsed_url = urlparse(url)
            original_name = os.path.basename(parsed_url.path) or "downloaded_file"
            tmp_path = Path(tmp_dir) / original_name

            with self.session.get(url, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()

                # 1. MIME CHECK
                content_type = r.headers.get("Content-Type", "").split(";")[0].lower()
                if content_type not in ALLOWED_MIMES:
                    raise ValueError(
                        f"Unsupported MIME type: {content_type} from {url}"
                    )

                # 2. Filename Refinement (Content-Disposition)
                content_disp = r.headers.get("Content-Disposition")
                if content_disp and "filename=" in content_disp:
                    filename_part = content_disp.split("filename=")[1].strip("\"'")
                    tmp_path = Path(tmp_dir) / filename_part

                # 3. Stream to Disk
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)

            # Pass the canonical_url to the callback instead of the raw one
            return callback(tmp_path, source_url=canonical_url, **kwargs)

    @staticmethod
    def normalize_url(url: str) -> str:
        """Removes query parameters and fragments to get the 'canonical' source."""
        parsed = urlparse(url)
        # Reconstruct without query (?) or fragment (#)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

    def get_head_info(self, url: str) -> Dict[str, Any]:
        """
        Performs a HEAD request to gather metadata without downloading the body.
        Returns a dict with etag, size, and mime.
        """
        try:
            with self.session.head(
                url, timeout=self.timeout, allow_redirects=True
            ) as r:
                r.raise_for_status()
                return {
                    "etag": r.headers.get("ETag", "").strip('"'),
                    "size": int(r.headers.get("Content-Length", 0)),
                    "mime": r.headers.get("Content-Type", "").split(";")[0].lower(),
                    "last_modified": r.headers.get("Last-Modified"),
                }
        except Exception as e:
            print(f"[{self.agent_id}] Head check failed for {url}: {e}")
            return {}
