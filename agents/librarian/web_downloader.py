import time
import requests
import tempfile
from pathlib import Path
from typing import Callable, Any, Optional
from functools import wraps
from urllib.parse import urlparse
import os


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
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Extract the filename from the URL
            parsed_url = urlparse(url)
            original_name = os.path.basename(parsed_url.path)

            # Fallback if URL is empty or weird (e.g., https://site.com/)
            if not original_name or "." not in original_name:
                original_name = "downloaded_file.bin"

            tmp_path = Path(tmp_dir) / original_name

            with self.session.get(url, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()

                # OPTIONAL: Check Content-Disposition header for a "real" filename
                # Some servers serve 'download.php?id=123' but header says 'manual.pdf'
                content_disp = r.headers.get("Content-Disposition")
                if content_disp and "filename=" in content_disp:
                    # Very basic parsing of 'attachment; filename="example.pdf"'
                    filename_part = content_disp.split("filename=")[1].strip("\"'")
                    tmp_path = Path(tmp_dir) / filename_part

                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)

            # Pass the correctly named temp file to the Agent
            return callback(tmp_path, source_url=url, **kwargs)
