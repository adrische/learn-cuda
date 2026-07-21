# modal run run_triton.py

import modal
app = modal.App("test-triton")

image = (
    modal.Image.debian_slim()
    .uv_pip_install("torch", "triton", "matplotlib", "pandas", "tabulate", "einops", "jaxtyping", "triton")
    .add_local_python_source("fused_softmax")   
)

volume = modal.Volume.from_name("test", create_if_missing=True)

@app.function(image=image, gpu="A100-80GB", volumes={"/data": volume})
def main():
    from fused_softmax import test, benchmark_run
    
    test()
    
    from pathlib import Path
    output_dir = Path("/data/print_path/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    benchmark_run(print_data=True, save_path=output_dir)
    volume.commit()

# run programatically, e.g., from notebook
# with modal.enable_output():
#     with app.run():
#         run.remote()