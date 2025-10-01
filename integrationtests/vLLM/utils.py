import os
import subprocess
import time
import requests
import sys
from typing import Optional

class VLLMServer:
    def __init__(
        self,
        model_name: str,
        server_args: Optional[list[str]],
        server_envs: Optional[dict[str, str]] = None,
        port: int = 8000,
        timeout: int = 240,
    ):
        cmd = [
            "vllm", "serve",
            model_name,
            "--port", str(port)
        ] + server_args

        env = os.environ.copy()
        if server_envs:
            env.update(server_envs)
        self.proc = subprocess.Popen(cmd,
                                    stdout=sys.stdout,
                                    stderr=sys.stderr,
                                    env=env
                                    )

        # Wait for server to be ready
        url = f"http://localhost:{port}/health"
        start_time = time.time()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    break
            except Exception:
                if time.time() - start_time > timeout:
                    # If still running let's kill the process
                    self._kill_proc()
                    raise TimeoutError("vLLM server did not start within timeout.")
            time.sleep(1)

    def _kill_proc(self):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
                print("vLLM server terminated.")
            except subprocess.TimeoutExpired:
                self.proc.kill()
                print("vLLM server forcefully killed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._kill_proc()