# modal run test_modal.py

# Modified from
# https://github.com/gpu-mode/reference-kernels/pull/96
# need the files corresponding to the problem listed in add_local_python_source in the PATH

import modal

from pathlib import Path

import sys
sys.path.insert(0, Path(__file__).parent)


image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .uv_pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("numpy")
    .uv_pip_install("ninja")
    .add_local_python_source("reference", "task", "utils", "submission")
)
app = modal.App("debug", image=image)


@app.function(gpu="A100-80GB")
def run(task_config: dict):
    import torch
    sys.path.insert(0, Path(__file__).parent)

    # from reference import generate_input, ref_kernel
    from reference import generate_input
    from submission import custom_kernel

    def clear_l2_cache():
        # import cupy as cp
        # cp.cuda.runtime.deviceSetLimit(cp.cuda.runtime.cudaLimitPersistingL2CacheSize, 0)
        # create a large dummy tensor
        dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda")
        # write stuff to
        dummy.fill_(42)
        del dummy


    for cfg in task_config["benchmarks"]:

        import time
        time.sleep(0.1) # I thought this could help stabilize performance consistency
        data = generate_input(**cfg)
        
        # https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp_v2/eval.py#L264
        durations = []
        for reps in range(20):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            clear_l2_cache()

            start_event.record()
            output = custom_kernel(data)
            end_event.record()
            torch.cuda.synchronize()
            duration = start_event.elapsed_time(end_event) * 1e3  # Convert ms to mus
            durations.append(duration)
        
        print(cfg)
        print(
            f"  duration (mus): {sum(durations)/len(durations)}"
        )


@app.local_entrypoint()
def main():
    import yaml

    task_yaml = Path(__file__).parent / "task.yml"
    task_config = yaml.safe_load(open(task_yaml))
    run.remote(task_config)