"""
Execution orchestrator.
=======================
Ties the sandbox to challenge definitions — the single entry point
for running or submitting user code.
"""

from dataclasses import dataclass
from typing import Optional

from challenges.registry import ChallengeRegistry
from engine.sandbox import run_in_sandbox


@dataclass
class ExecutionResult:
    """Combined result of sandbox execution + optional grading."""

    success: bool
    mode: str
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0
    error: Optional[str] = None
    traceback_str: Optional[str] = None
    timed_out: bool = False

    # Run-mode extras
    sample_accuracy: Optional[float] = None
    train_time: Optional[float] = None

    # Submit-mode extras
    grading: Optional[dict] = None

    def to_dict(self) -> dict:
        d: dict = {
            "success": self.success,
            "mode": self.mode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "timed_out": self.timed_out,
        }
        if self.error:
            d["error"] = self.error
        if self.traceback_str:
            d["traceback"] = self.traceback_str
        if self.sample_accuracy is not None:
            d["sample_accuracy"] = self.sample_accuracy
        if self.train_time is not None:
            d["train_time"] = self.train_time
        if self.grading is not None:
            d["grading"] = self.grading
        return d


def execute_challenge(
    challenge_id: str,
    user_code: str,
    mode: str = "run",
) -> ExecutionResult:
    """
    Execute user code for a challenge.

    Parameters
    ----------
    challenge_id : str
    user_code : str
    mode : ``'run'`` (quick test) or ``'submit'`` (full grading).

    Returns
    -------
    ExecutionResult
    """
    registry = ChallengeRegistry()
    challenge = registry.get(challenge_id)

    if challenge is None:
        return ExecutionResult(
            success=False,
            mode=mode,
            error=f"Challenge not found: {challenge_id}",
        )

    timeout = challenge.run_timeout if mode == "run" else challenge.timeout

    # ── Run in sandbox ───────────────────────────────────────
    sandbox = run_in_sandbox(
        user_code=user_code,
        harness_code=challenge.get_harness_code(mode),
        harness_args=challenge.get_harness_args(mode),
        data_files=challenge.get_data_files(mode),
        timeout=timeout,
    )

    output = sandbox.output_data or {}

    result = ExecutionResult(
        success=sandbox.success,
        mode=mode,
        stdout=sandbox.stdout,
        stderr=sandbox.stderr,
        execution_time=sandbox.execution_time,
        timed_out=sandbox.timed_out,
        error=output.get("error"),
        traceback_str=output.get("traceback_str"),
        sample_accuracy=output.get("sample_accuracy"),
        train_time=output.get("train_time"),
    )

    # ── Grade on submit ──────────────────────────────────────
    if mode == "submit" and sandbox.success:
        result.grading = challenge.grade(
            user_code=user_code,
            execution_output=output,
        )

    return result
