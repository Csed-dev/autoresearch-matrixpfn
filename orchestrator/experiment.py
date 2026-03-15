from __future__ import annotations

import datetime
import re
from pathlib import Path

from orchestrator.errors import (
    ExperimentNotFoundError,
    ExperimentNotReadyError,
    GpuUnavailableError,
    PodPoolFullError,
)
from orchestrator.models import (
    ExperimentSpec,
    ExperimentState,
    ExperimentStatus,
    SchedulerConfig,
    Template,
    load_experiments,
    load_state,
    save_experiment_config,
    save_experiment_result,
    save_state,
)
from orchestrator.pod import PodConnection, PodManager


class ExperimentManager:
    def __init__(self, experiments_path: str, config_path: str, state_path: str = "scheduler_state.json"):
        self._config = SchedulerConfig.from_yaml(config_path)
        self._templates, self._specs = load_experiments(experiments_path)
        self._spec_map = {s.name: s for s in self._specs}
        self._state_path = Path(state_path)
        self._results_dir = Path(self._config.results_dir)
        self._pm = PodManager()

        self._states: dict[str, ExperimentState] = {}
        if self._state_path.exists():
            self._states = load_state(self._state_path)

        for spec in self._specs:
            if spec.name not in self._states:
                self._states[spec.name] = ExperimentState(name=spec.name)

        self._active_pods: dict[str, PodConnection] = {}

    def validate(self) -> str:
        issues = []
        for spec in self._specs:
            for dep in spec.dependencies:
                if dep not in self._spec_map:
                    issues.append(f"  {spec.name}: dependency '{dep}' not found")
        if issues:
            return "Validation FAILED:\n" + "\n".join(issues)
        return f"Validation OK: {len(self._specs)} experiments, {len(self._templates)} templates, no issues."

    def status(self) -> str:
        self._propagate_blocked()
        lines = ["Experiment Status:"]
        for spec in self._specs:
            state = self._states[spec.name]
            deps_str = f" (deps: {', '.join(spec.dependencies)})" if spec.dependencies else ""
            gpu = spec.gpu_type or self._config.gpu_type
            desc = f" — {spec.description.strip()[:60]}" if spec.description else ""
            lines.append(f"  {spec.name:30s} {state.status.value:10s} gpu={gpu}{deps_str}{desc}")
        running = [n for n, s in self._states.items() if s.status == ExperimentStatus.RUNNING]
        ready = self.get_ready()
        lines.append(f"\nRunning: {len(running)}/{self._config.max_pods}  Ready: {len(ready)}  "
                      f"Completed: {sum(1 for s in self._states.values() if s.status == ExperimentStatus.COMPLETED)}  "
                      f"Failed: {sum(1 for s in self._states.values() if s.status == ExperimentStatus.FAILED)}  "
                      f"Blocked: {sum(1 for s in self._states.values() if s.status == ExperimentStatus.BLOCKED)}")
        return "\n".join(lines)

    def get_ready(self) -> list[str]:
        self._propagate_blocked()
        ready = []
        for spec in self._specs:
            state = self._states[spec.name]
            if state.status != ExperimentStatus.PENDING:
                continue
            dep_states = [self._states[d].status for d in spec.dependencies]
            if all(s == ExperimentStatus.COMPLETED for s in dep_states):
                ready.append(spec.name)
        return ready

    def start(self, experiment_name: str, gpu_type: str | None = None) -> tuple[str, str]:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))

        state = self._states[experiment_name]
        if state.status not in (ExperimentStatus.PENDING, ExperimentStatus.READY):
            ready = self.get_ready()
            waiting = [d for d in self._spec_map[experiment_name].dependencies
                       if self._states[d].status != ExperimentStatus.COMPLETED]
            raise ExperimentNotReadyError(experiment_name, state.status.value, waiting)

        if experiment_name not in self.get_ready():
            spec = self._spec_map[experiment_name]
            waiting = [d for d in spec.dependencies
                       if self._states[d].status != ExperimentStatus.COMPLETED]
            raise ExperimentNotReadyError(experiment_name, state.status.value, waiting)

        running_count = sum(1 for s in self._states.values() if s.status == ExperimentStatus.RUNNING)
        if running_count >= self._config.max_pods:
            running_names = [n for n, s in self._states.items() if s.status == ExperimentStatus.RUNNING]
            raise PodPoolFullError(self._config.max_pods, running_names)

        spec = self._spec_map[experiment_name]
        effective_gpu = gpu_type or spec.gpu_type or self._config.gpu_type

        try:
            pod_id = self._pm.create_pod(f"exp-{experiment_name}", gpu_type=effective_gpu)
        except Exception as e:
            if "no longer any instances" in str(e).lower() or "does not have the resources" in str(e).lower():
                gpus = self._pm.get_available_gpus()
                top5 = "\n".join(f"  {g['id']:40s} {g['memory_gb']:>4}GB  ${g['price_per_hr']:.2f}/hr" for g in gpus[:5])
                raise GpuUnavailableError(effective_gpu, f"Cheapest available:\n{top5}") from e
            raise

        state.status = ExperimentStatus.RUNNING
        state.pod_id = pod_id
        state.started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._persist()

        save_experiment_config(self._results_dir, spec, self._config)

        conn = self._pm.wait_until_ready(pod_id)
        self._active_pods[pod_id] = conn

        workspace = self._config.workspace_dir
        self._pm.ssh_run(conn, self._config.setup_command.format(workspace=workspace),
                         timeout=self._config.setup_timeout)

        env_export = " ".join(f"export {k}={_shell_quote(v)};" for k, v in spec.env.items())
        command = spec.command.format(workspace=workspace)
        full_command = f"{env_export} {command}" if env_export else command
        timeout = spec.timeout or self._config.experiment_timeout

        ssh_command = self._build_ssh_command(conn, full_command, timeout)
        return pod_id, ssh_command

    def finish(self, experiment_name: str, exit_code: int, output: str) -> None:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))

        state = self._states[experiment_name]
        state.finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state.exit_code = exit_code

        if exit_code == 0:
            state.status = ExperimentStatus.COMPLETED
        else:
            state.status = ExperimentStatus.FAILED
            state.error = _extract_error(output)

        save_experiment_result(self._results_dir, experiment_name, state, output)

        if state.pod_id:
            self._pm.terminate_pod(state.pod_id)
            self._active_pods.pop(state.pod_id, None)

        self._propagate_blocked()
        self._persist()

    def result(self, experiment_name: str) -> str:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))

        state = self._states[experiment_name]
        spec = self._spec_map[experiment_name]
        exp_dir = self._results_dir / experiment_name
        lines = [f"Experiment: {experiment_name}", f"Status: {state.status.value}"]
        if spec.description:
            lines.append(f"Description: {spec.description.strip()}")
        if spec.hypothesis:
            lines.append(f"Hypothesis: {spec.hypothesis.strip()}")

        if state.started_at:
            lines.append(f"Started: {state.started_at}")
        if state.finished_at:
            lines.append(f"Finished: {state.finished_at}")
        if state.exit_code is not None:
            lines.append(f"Exit code: {state.exit_code}")
        if state.error:
            lines.append(f"Error: {state.error}")

        log_path = exp_dir / "log.txt"
        if log_path.exists():
            log = log_path.read_text()
            metrics = _extract_metrics(log)
            if metrics:
                lines.append("\nMetrics:")
                for k, v in metrics.items():
                    lines.append(f"  {k}: {v}")
            lines.append(f"\nLog (last 30 lines):")
            for line in log.strip().split("\n")[-30:]:
                lines.append(f"  {line}")

        return "\n".join(lines)

    def available_templates(self) -> str:
        lines = ["Available templates:"]
        for name, tpl in self._templates.items():
            gpu = tpl.gpu_type or self._config.gpu_type
            env_keys = ", ".join(tpl.env.keys()) if tpl.env else "none"
            lines.append(f"  {name:20s} gpu={gpu}  env=[{env_keys}]")
        return "\n".join(lines)

    def available_experiments(self) -> str:
        return self.status()

    def available_gpus(self, min_memory_gb: int = 20) -> str:
        gpus = self._pm.get_available_gpus(min_memory_gb)
        lines = ["Available GPUs:"]
        for g in gpus:
            lines.append(f"  {g['id']:45s} {g['memory_gb']:>4}GB  ${g['price_per_hr']:.2f}/hr")
        return "\n".join(lines)

    def cleanup(self) -> str:
        pods = self._pm.list_pods()
        running_pod_ids = {s.pod_id for s in self._states.values()
                          if s.status == ExperimentStatus.RUNNING and s.pod_id}
        orphans = [p for p in pods if p.status in ("RUNNING", "STARTING")
                   and p.pod_id not in running_pod_ids]
        if not orphans:
            return "No orphaned pods found."
        lines = ["Terminating orphaned pods:"]
        for pod in orphans:
            self._pm.terminate_pod(pod.pod_id)
            lines.append(f"  {pod.pod_id} | {pod.name} | {pod.gpu_type} — terminated")
        return "\n".join(lines)

    def reset(self, experiment_name: str) -> str:
        if experiment_name not in self._spec_map:
            raise ExperimentNotFoundError(experiment_name, list(self._spec_map.keys()))
        state = self._states[experiment_name]
        if state.status == ExperimentStatus.RUNNING and state.pod_id:
            self._pm.terminate_pod(state.pod_id)
        state.status = ExperimentStatus.PENDING
        state.pod_id = None
        state.started_at = None
        state.finished_at = None
        state.exit_code = None
        state.error = None
        self._persist()
        return f"Experiment '{experiment_name}' reset to PENDING."

    def _propagate_blocked(self) -> None:
        changed = True
        while changed:
            changed = False
            for spec in self._specs:
                state = self._states[spec.name]
                if state.status != ExperimentStatus.PENDING:
                    continue
                for dep in spec.dependencies:
                    dep_status = self._states[dep].status
                    if dep_status in (ExperimentStatus.FAILED, ExperimentStatus.BLOCKED):
                        state.status = ExperimentStatus.BLOCKED
                        state.error = f"Blocked by failed dependency: {dep}"
                        changed = True
                        break

    def _persist(self) -> None:
        save_state(self._state_path, self._states)

    def _build_ssh_command(self, conn: PodConnection, command: str, timeout: int) -> str:
        return (
            f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "
            f"-p {conn.port} root@{conn.ip} "
            f"'timeout {timeout} bash -c {_shell_quote(command)}'"
        )


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _extract_error(output: str) -> str:
    lines = output.strip().split("\n")
    error_lines = [l for l in lines[-20:] if "error" in l.lower() or "traceback" in l.lower() or "exception" in l.lower()]
    if error_lines:
        return error_lines[-1][:200]
    return lines[-1][:200] if lines else "Unknown error"


def _extract_metrics(log: str) -> dict[str, str]:
    metrics = {}
    patterns = [
        "score", "synthetic_score", "suitesparse_score",
        "synthetic_conv", "suitesparse_conv", "peak_vram_mb",
        "num_epochs", "num_params", "best_loss", "total_seconds",
    ]
    for key in patterns:
        match = re.search(rf"^{re.escape(key)}:\s+(.+)$", log, re.MULTILINE)
        if match:
            metrics[key] = match.group(1).strip()
    return metrics
