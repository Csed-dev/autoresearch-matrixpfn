import re
from dataclasses import dataclass

from orchestrator.config import (
    EXPERIMENT_TIMEOUT_S,
    SETUP_TIMEOUT_S,
    WORKSPACE_DIR,
)
from orchestrator.pod import PodConnection, PodManager


@dataclass(frozen=True)
class ExperimentResult:
    score: float
    synthetic_score: float
    suitesparse_score: float
    synthetic_conv_pct: float
    suitesparse_conv_pct: float
    peak_vram_mb: float
    num_epochs: int
    num_params: int
    best_loss: float
    training_seconds: float
    total_seconds: float
    raw_log: str


class ExperimentRunner:
    def __init__(self, pod_manager: PodManager, connection: PodConnection):
        self._pm = pod_manager
        self._conn = connection

    def setup_pod(self, branch: str = "main") -> None:
        setup_script = self._pm.ssh_run(
            self._conn,
            f"cat {WORKSPACE_DIR}/setup_pod.sh 2>/dev/null || echo '__MISSING__'",
            timeout=10,
        )
        if "__MISSING__" in setup_script:
            self._pm.ssh_run(
                self._conn,
                f"git clone https://github.com/Csed-dev/autoresearch-matrixpfn.git {WORKSPACE_DIR}",
                timeout=SETUP_TIMEOUT_S,
            )
        self._pm.ssh_run(
            self._conn,
            f"bash {WORKSPACE_DIR}/setup_pod.sh {branch}",
            timeout=SETUP_TIMEOUT_S,
        )

    def sync_code(self, branch: str = "main") -> None:
        self._pm.ssh_run(
            self._conn,
            f"cd {WORKSPACE_DIR} && git fetch origin && git checkout {branch} && git reset --hard origin/{branch}",
            timeout=60,
        )

    def run_experiment(self) -> ExperimentResult:
        log = self._pm.ssh_run(
            self._conn,
            f"cd {WORKSPACE_DIR} && uv run train.py 2>&1",
            timeout=EXPERIMENT_TIMEOUT_S,
        )
        return _parse_log(log)

    def retrieve_file(self, remote_path: str) -> str:
        return self._pm.ssh_run(
            self._conn,
            f"cat {WORKSPACE_DIR}/{remote_path}",
            timeout=30,
        )


def _parse_log(log: str) -> ExperimentResult:
    return ExperimentResult(
        score=_extract_float(log, "score"),
        synthetic_score=_extract_float(log, "synthetic_score"),
        suitesparse_score=_extract_float(log, "suitesparse_score"),
        synthetic_conv_pct=_extract_float(log, "synthetic_conv"),
        suitesparse_conv_pct=_extract_float(log, "suitesparse_conv"),
        peak_vram_mb=_extract_float(log, "peak_vram_mb"),
        num_epochs=int(_extract_float(log, "num_epochs")),
        num_params=int(_extract_float(log, "num_params")),
        best_loss=_extract_float(log, "best_loss"),
        training_seconds=_extract_float(log, "training_seconds"),
        total_seconds=_extract_float(log, "total_seconds"),
        raw_log=log,
    )


def _extract_float(log: str, key: str) -> float:
    pattern = rf"^{re.escape(key)}:\s+([0-9.eE+\-]+)"
    match = re.search(pattern, log, re.MULTILINE)
    if not match:
        raise ValueError(f"Key '{key}' not found in experiment log")
    return float(match.group(1))
