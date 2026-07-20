# modal run run_triton.py

import modal
app = modal.App("test-triton")

image = (
    modal.Image.debian_slim()
    .uv_pip_install("torch", "triton", "matplotlib", "pandas", "tabulate", "einops", "jaxtyping", "triton")
    .add_local_python_source("weighted_sum")   
)


@app.function(image=image, gpu="A100-40GB")
def main():
    from weighted_sum import test
    
    test() # tl.make_block_ptr is deprecated

# run programatically, e.g., from notebook
# with modal.enable_output():
#     with app.run():
#         run.remote()