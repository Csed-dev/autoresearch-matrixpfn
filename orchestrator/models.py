from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from enum import Enum
from graphlib import TopologicalSorter, CycleError
from pathlib import Path

import yaml

from orchestrator.errors import (
    CyclicDependencyError,
    DependencyNotFoundError,
    TemplateNotFoundError,
)


class ExperimentStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Template:
    name: str
    command: str
    env: dict[str, str] = field(default_factory=dict)
    gpu_type: str | None = None
    timeout: int | None = None


@dataclass
class ExperimentSpec:
    name: str
    command: str
    description: str = ""
    hypothesis: str = ""
    env: dict[str, str] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    gpu_type: str | None = None
    timeout: int | None = None
    template_name: str | None = None


@dataclass
class ExperimentState:
    name: str
    status: ExperimentStatus = ExperimentStatus.PENDING
    pod_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None


@dataclass
class SchedulerConfig:
    max_pods: int
    gpu_type: str
    image: str
    container_disk_gb: int
    repo_url: str
    workspace_dir: str
    setup_command: str
    sync_command: str
    experiment_timeout: int
    setup_timeout: int
    pod_ready_timeout: int
    log_dir: str
    results_dir: str
    poll_interval: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> SchedulerConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            max_pods=data["max_pods"],
            gpu_type=data["gpu_type"],
            image=data["image"],
            container_disk_gb=data.get("container_disk_gb", 20),
            repo_url=data["repo_url"],
            workspace_dir=data["workspace_dir"],
            setup_command=data["setup_command"],
            sync_command=data["sync_command"],
            experiment_timeout=data.get("experiment_timeout", 900),
            setup_timeout=data.get("setup_timeout", 600),
            pod_ready_timeout=data.get("pod_ready_timeout", 120),
            log_dir=data.get("log_dir", "logs"),
            results_dir=data.get("results_dir", "results"),
            poll_interval=data.get("poll_interval", 10),
        )


def load_experiments(path: str | Path) -> tuple[dict[str, Template], list[ExperimentSpec]]:
    with open(path) as f:
        data = yaml.safe_load(f)

    templates = {}
    for name, tpl_data in data.get("templates", {}).items():
        templates[name] = Template(
            name=name,
            command=tpl_data["command"],
            env=tpl_data.get("env", {}),
            gpu_type=tpl_data.get("gpu_type"),
            timeout=tpl_data.get("timeout"),
        )

    experiment_names = [e["name"] for e in data.get("experiments", [])]

    experiments = []
    for exp_data in data.get("experiments", []):
        template_name = exp_data.get("template")
        if template_name and template_name not in templates:
            raise TemplateNotFoundError(template_name, list(templates.keys()))

        if template_name:
            tpl = templates[template_name]
            command = exp_data.get("command", tpl.command)
            env = {**tpl.env, **exp_data.get("env", {})}
            gpu_type = exp_data.get("gpu_type", tpl.gpu_type)
            timeout = exp_data.get("timeout", tpl.timeout)
        else:
            command = exp_data["command"]
            env = exp_data.get("env", {})
            gpu_type = exp_data.get("gpu_type")
            timeout = exp_data.get("timeout")

        deps = exp_data.get("dependencies", [])
        for dep in deps:
            if dep not in experiment_names:
                raise DependencyNotFoundError(exp_data["name"], dep, experiment_names)

        experiments.append(ExperimentSpec(
            name=exp_data["name"],
            command=command,
            description=exp_data.get("description", ""),
            hypothesis=exp_data.get("hypothesis", ""),
            env=env,
            dependencies=deps,
            gpu_type=gpu_type,
            timeout=timeout,
            template_name=template_name,
        ))

    _validate_dag(experiments)
    return templates, experiments


def _validate_dag(experiments: list[ExperimentSpec]) -> None:
    graph = {}
    for exp in experiments:
        graph[exp.name] = set(exp.dependencies)
    try:
        ts = TopologicalSorter(graph)
        ts.prepare()
    except CycleError as e:
        raise CyclicDependencyError(str(e)) from e


def _get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def save_experiment_config(results_dir: Path, spec: ExperimentSpec, config: SchedulerConfig) -> None:
    exp_dir = results_dir / spec.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "name": spec.name,
        "description": spec.description,
        "hypothesis": spec.hypothesis,
        "template": spec.template_name,
        "command": spec.command,
        "env": spec.env,
        "dependencies": spec.dependencies,
        "gpu_type": spec.gpu_type or config.gpu_type,
        "timeout": spec.timeout or config.experiment_timeout,
        "image": config.image,
        "git_commit": _get_git_commit(),
    }
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False)


def save_experiment_result(results_dir: Path, name: str, state: ExperimentState, log: str) -> None:
    exp_dir = results_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "log.txt", "w") as f:
        f.write(log)
    status_data = {
        "name": state.name,
        "status": state.status.value,
        "pod_id": state.pod_id,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "exit_code": state.exit_code,
        "error": state.error,
    }
    with open(exp_dir / "status.json", "w") as f:
        json.dump(status_data, f, indent=2)


def save_state(path: Path, experiments: dict[str, ExperimentState]) -> None:
    data = {}
    for name, state in experiments.items():
        data[name] = {
            "status": state.status.value,
            "pod_id": state.pod_id,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "exit_code": state.exit_code,
            "error": state.error,
        }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_state(path: Path) -> dict[str, ExperimentState]:
    with open(path) as f:
        data = json.load(f)
    states = {}
    for name, s in data.items():
        states[name] = ExperimentState(
            name=name,
            status=ExperimentStatus(s["status"]),
            pod_id=s.get("pod_id"),
            started_at=s.get("started_at"),
            finished_at=s.get("finished_at"),
            exit_code=s.get("exit_code"),
            error=s.get("error"),
        )
    return states
