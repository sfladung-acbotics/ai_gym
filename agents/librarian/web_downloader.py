import time
import requests
import tempfile
from pathlib import Path
from typing import Callable, Any, Optional
from functools import wraps


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
        """
        Downloads a file and executes a callback.
        The TemporaryDirectory is deleted immediately after the block finishes.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "download_buffer"

            with self.session.get(url, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)

            # The file exists here. Once this callback returns,
            # the 'with' block ends and the file is nuked.
            return callback(tmp_path, source_url=url, **kwargs)
