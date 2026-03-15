class OrchestratorError(Exception):
    pass


class TemplateNotFoundError(OrchestratorError):
    def __init__(self, template_name: str, available: list[str]):
        self.template_name = template_name
        self.available = available
        super().__init__(
            f"Template '{template_name}' not found.\n"
            f"Available templates: {', '.join(available)}"
        )


class ExperimentNotFoundError(OrchestratorError):
    def __init__(self, experiment_name: str, available: list[str]):
        self.experiment_name = experiment_name
        self.available = available
        candidates = [n for n in available if n.startswith(experiment_name[:3])]
        msg = f"Experiment '{experiment_name}' not found.\nDefined experiments: {', '.join(available)}"
        if candidates:
            msg += f"\nDid you mean: {', '.join(candidates)}?"
        super().__init__(msg)


class DependencyNotFoundError(OrchestratorError):
    def __init__(self, experiment_name: str, missing_dep: str, available: list[str]):
        super().__init__(
            f"Experiment '{experiment_name}' depends on '{missing_dep}' which doesn't exist.\n"
            f"Defined experiments: {', '.join(available)}"
        )


class CyclicDependencyError(OrchestratorError):
    def __init__(self, cycle_info: str):
        super().__init__(f"Cyclic dependency detected: {cycle_info}")


class ExperimentNotReadyError(OrchestratorError):
    def __init__(self, experiment_name: str, status: str, waiting_on: list[str]):
        waiting_str = ", ".join(waiting_on) if waiting_on else "nothing"
        super().__init__(
            f"Experiment '{experiment_name}' is not ready.\n"
            f"Status: {status}. Waiting on: [{waiting_str}]"
        )


class PodPoolFullError(OrchestratorError):
    def __init__(self, max_pods: int, running: list[str]):
        super().__init__(
            f"Pod pool is full ({max_pods}/{max_pods}).\n"
            f"Running experiments: {', '.join(running)}\n"
            f"Wait for one to finish or increase max_pods in scheduler_config.yaml"
        )


class GpuUnavailableError(OrchestratorError):
    def __init__(self, gpu_type: str, available_hint: str):
        super().__init__(
            f"GPU '{gpu_type}' unavailable.\n"
            f"{available_hint}\n"
            f"Run: mgr.available_gpus() for full list."
        )
