import sys

sys.path.insert(0, ".")

from orchestrator import PodManager, ExperimentRunner


def test_pod_lifecycle():
    pm = PodManager()

    print("1. Checking for orphaned pods...")
    for pod in pm.list_pods():
        print(f"   {pod.pod_id} | {pod.name} | {pod.status} | {pod.gpu_type} | ${pod.cost_per_hr}/hr")

    print("2. Creating pod...")
    pod_id = pm.create_pod("test-lifecycle")
    print(f"   Pod ID: {pod_id}")

    try:
        print("3. Waiting for pod to be ready...")
        conn = pm.wait_until_ready(pod_id)
        print(f"   Connected: {conn.ip}:{conn.port}")

        print("4. Testing SSH (nvidia-smi)...")
        gpu_info = pm.ssh_run(conn, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
        print(f"   GPU: {gpu_info.strip()}")

        print("5. Testing SSH (python version)...")
        python_version = pm.ssh_run(conn, "python3 --version")
        print(f"   {python_version.strip()}")

        print("6. Testing SSH (disk space)...")
        disk = pm.ssh_run(conn, "df -h / | tail -1")
        print(f"   Disk: {disk.strip()}")

        print("ALL TESTS PASSED")
    finally:
        print("7. Terminating pod...")
        pm.terminate_pod(pod_id)
        print("   Pod terminated")


def test_experiment_setup():
    pm = PodManager()

    print("1. Creating pod...")
    pod_id = pm.create_pod("test-setup")

    try:
        print("2. Waiting for pod...")
        conn = pm.wait_until_ready(pod_id)
        print(f"   Connected: {conn.ip}:{conn.port}")

        runner = ExperimentRunner(pm, conn)

        print("3. Setting up pod (git clone, uv sync, prepare.py)...")
        runner.setup_pod(branch="main")
        print("   Setup complete")

        print("4. Verifying repo...")
        files = pm.ssh_run(conn, "ls /workspace/autoresearch-matrixpfn/train.py")
        print(f"   {files.strip()}")

        print("5. Verifying uv venv...")
        uv_check = pm.ssh_run(conn, "cd /workspace/autoresearch-matrixpfn && uv run python3 -c 'import matrixpfn; print(matrixpfn.__version__)'")
        print(f"   matrixpfn version: {uv_check.strip()}")

        print("ALL TESTS PASSED")
    finally:
        print("6. Terminating pod...")
        pm.terminate_pod(pod_id)
        print("   Pod terminated")


def test_full_experiment():
    pm = PodManager()

    print("1. Creating pod...")
    pod_id = pm.create_pod("test-full")

    try:
        print("2. Waiting for pod...")
        conn = pm.wait_until_ready(pod_id)
        print(f"   Connected: {conn.ip}:{conn.port}")

        runner = ExperimentRunner(pm, conn)

        print("3. Setting up pod...")
        runner.setup_pod(branch="main")
        print("   Setup complete")

        print("4. Running experiment (~7 min)...")
        result = runner.run_experiment()

        print(f"   score:             {result.score:.6f}")
        print(f"   synthetic_score:   {result.synthetic_score:.6f}")
        print(f"   suitesparse_score: {result.suitesparse_score:.6f}")
        print(f"   synthetic_conv:    {result.synthetic_conv_pct:.1f}%")
        print(f"   suitesparse_conv:  {result.suitesparse_conv_pct:.1f}%")
        print(f"   peak_vram_mb:      {result.peak_vram_mb:.1f}")
        print(f"   num_epochs:        {result.num_epochs}")
        print(f"   num_params:        {result.num_params}")
        print(f"   total_seconds:     {result.total_seconds:.1f}")

        assert result.score > 0, "Score must be positive"
        assert result.score < 1.0, "Score must be less than 1.0"
        assert result.num_epochs > 0, "Must have trained at least one epoch"

        print("ALL TESTS PASSED")
    finally:
        print("5. Terminating pod...")
        pm.terminate_pod(pod_id)
        print("   Pod terminated")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test",
        choices=["lifecycle", "setup", "full"],
        help="lifecycle: pod create/ssh/terminate. setup: git clone + deps. full: run experiment.",
    )
    args = parser.parse_args()

    {"lifecycle": test_pod_lifecycle, "setup": test_experiment_setup, "full": test_full_experiment}[args.test]()
