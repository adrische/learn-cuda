import modal
# modal setup
# modal run script.py

image = (
    modal.Image.debian_slim()
    .uv_pip_install("torch", "triton", "matplotlib", "pandas", "tabulate")
)

with image.imports():
    import torch
    import triton
    import triton.language as tl

app = modal.App("triton-tests")

@app.function(gpu="A100-40GB", image=image)
def main():
    
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    
    @triton.jit()
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, # like Cuda
                   BLOCK_SIZE: tl.constexpr):
        # int idx = threadIdx.x + blockIdx.x*blockDim.x;
        pid = tl.program_id(axis=0) # blockIdx.x
        block_start = pid * BLOCK_SIZE # blockIdx.x*blockDim.x
        offsets = block_start + tl.arange(0, BLOCK_SIZE) # all idx within a block
        mask = offsets < n_elements # if (idx <  N)
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    
    
    def add(x: torch.Tensor, y: torch.Tensor):
        output = torch.empty_like(x)
        # assert all(getattr(vec, "device") == DEVICE for vec in [x, y, output])
        assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output
    
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device = DEVICE)
    y = torch.rand(size, device = DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch[:3])
    print(output_triton[:3])
    max_diff = torch.max(torch.abs(output_torch - output_triton))
    print(f"{max_diff=}")
    
    @triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
    def benchmark(size, provider):
        x = torch.rand(size, device=DEVICE, dtype=torch.float32)
        y = torch.rand(size, device=DEVICE, dtype=torch.float32)
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
        gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
        return gbps(ms), gbps(max_ms), gbps(min_ms)

    benchmark.run(print_data=True)

# with gpu="A100-40GB"
# vector-add-performance:
#            size  Triton (GB/s)  Torch (GB/s)
# 0        4096.0       5.333333      5.333333
# 1        8192.0      12.000000     12.000000
# 2       16384.0      24.000000     27.428571
# 3       32768.0      54.857142     54.857142
# 4       65536.0      96.000000     96.000000
# 5      131072.0     170.666661    170.666661
# 6      262144.0     279.272725    307.200008
# 7      524288.0     472.615390    472.615390
# 8     1048576.0     722.823517    722.823517
# 9     2097152.0     945.230780    945.230780
# 10    4194304.0    1045.787204   1068.521715
# 11    8388608.0    1184.385557   1184.385557
# 12   16777216.0    1276.675375   1276.675375
# 13   33554432.0    1323.959634   1323.959634
# 14   67108864.0    1346.630084   1348.939930
# 15  134217728.0    1358.259103   1358.259103

