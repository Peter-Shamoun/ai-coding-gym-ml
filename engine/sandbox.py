"""
Sandboxed code execution engine.
================================
Runs user code in an isolated subprocess with timeout and output limits.

Current isolation: subprocess + timeout + temp directory.
For production security, wrap execution inside a Docker container.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SandboxResult:
    """Outcome of a sandboxed code execution."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    output_data: Optional[dict] = None
    execution_time: float = 0.0
    exit_code: int = -1
    timed_out: bool = False


def run_in_sandbox(
    user_code: str,
    harness_code: str,
    harness_args: list,
    data_files: Optional[dict] = None,
    timeout: int = 300,
    max_output: int = 1_048_576,
) -> SandboxResult:
    """
    Execute user code inside a sandboxed subprocess.

    Parameters
    ----------
    user_code : str
        The user's Python code (written to ``solution.py`` in sandbox).
    harness_code : str
        Harness script that imports and exercises the user's code.
    harness_args : list[str]
        CLI args for the harness.  ``output_path`` is appended automatically.
    data_files : dict[str, str] | None
        ``{sandbox_filename: source_absolute_path}`` — files to copy into the
        sandbox working directory.
    timeout : int
        Max execution time in seconds.
    max_output : int
        Max bytes of captured stdout / stderr.

    Returns
    -------
    SandboxResult
    """
    work_dir = tempfile.mkdtemp(prefix="gym_sandbox_")
    output_path = os.path.join(work_dir, "_output.json")

    try:
        # ── Write user code ──────────────────────────────────
        with open(os.path.join(work_dir, "solution.py"), "w", encoding="utf-8") as f:
            f.write(user_code)

        # ── Write harness ────────────────────────────────────
        harness_path = os.path.join(work_dir, "_harness.py")
        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(harness_code)

        # ── Copy data files ──────────────────────────────────
        if data_files:
            for filename, source in data_files.items():
                dest = os.path.join(work_dir, filename)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(source, dest)

        # ── Build command ────────────────────────────────────
        cmd = [sys.executable, "-u", harness_path, *harness_args, output_path]

        # ── Execute ──────────────────────────────────────────
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            elapsed = round(time.time() - start, 2)

            # Read structured output
            output_data = None
            if os.path.isfile(output_path):
                with open(output_path, "r") as f:
                    output_data = json.load(f)

            stdout = (proc.stdout or "")[:max_output]
            stderr = (proc.stderr or "")[:max_output]

            success = (
                proc.returncode == 0
                and output_data is not None
                and output_data.get("success", False)
            )

            return SandboxResult(
                success=success,
                stdout=stdout,
                stderr=stderr,
                output_data=output_data,
                execution_time=elapsed,
                exit_code=proc.returncode,
            )

        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - start, 2)
            return SandboxResult(
                success=False,
                stderr=f"Execution timed out after {timeout} seconds.",
                output_data={"success": False, "error": f"Timeout ({timeout}s)"},
                execution_time=elapsed,
                timed_out=True,
            )

    except Exception as exc:
        return SandboxResult(
            success=False,
            stderr=f"Sandbox setup error: {exc}",
            output_data={"success": False, "error": str(exc)},
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
