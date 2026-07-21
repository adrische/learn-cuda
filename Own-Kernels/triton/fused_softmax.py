import torch
import triton
import triton.language as tl
from triton.runtime import driver


DEVICE = triton.runtime.driver.active.get_active_torch_device()

# I won't be using AMD for now
# def is_hip():
#     return triton.runtime.driver.active.get_current_target().backend == "hip"

# def is_cdna():
#     return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
#                                                                                    'gfx90a', 'gfx908')


# try @torch.jit.script
def naive_softmax(x: torch.Tensor): # x: (r, c)
    x_max = x.max(dim=1).values # (r, ), alternatively: keepdim=True -> (r, 1)
    z = x - x_max[:, None] # x_max[:, None] (r, 1)
    numerator = z.exp() # (r, c)
    denominator = numerator.sum(dim=1) # (r, )
    ret = numerator / denominator[:, None]
    return ret

naive_softmax_compiled = torch.compile(naive_softmax)

@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride, # no col stride needed? -> usuilly col stride = 1 in torch
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr, # smallest integer larger or equal to n_cols
    num_stages: tl.constexpr
):
    row_start = tl.program_id(0) # which block
    row_step = tl.num_programs(0) # how many blocks in total
    # -> block-stride loop? each "program" processes one row
    for row_idx in tl.range(
        arg1=row_start, # start
        arg2=n_rows, # end
        step=row_step, # step
        num_stages=num_stages # load pipelining
        ):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE) # covers at least the entire row, but maybe more
        input_ptrs = row_start_ptr + col_offsets # could reach into the next row? why not input_col_stride?
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator # this is just one row, not a matrix-vector division as in Pytorch version
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


# Kernel launcher function
properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()


def softmax_triton(x: torch.Tensor):
    
    n_rows, n_cols = x.shape[0], x.shape[1]
    
    out = torch.empty_like(x)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Full occupancy calculation...
    # Left out the case for HIP
    num_stages = 4 if SIZE_SMEM > 200_000 else 2 # heuristic?
    num_warps = 8 # 8 * 32 = 256 threads
    kernel = softmax_kernel.warmup(out, x, x.stride(0), out.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages, num_warps=num_warps, grid=(1,))
    kernel._init_handles()
    n_regs = kernel.n_regs
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps) # n thread blocks per SMu?
    size_smem = kernel.metadata.shared # why not kernel.shared?
    occupancy = min(occupancy, SIZE_SMEM // size_smem) # was "warmed up" with 1 thread block, so SM per thread block
    num_programs = NUM_SM * occupancy # n thread blocks per device
    num_programs = min(num_programs, n_rows) # each thread block processes one row, don't need more than we have rows
    grid = (num_programs, 1, 1) # full grid
    
    softmax_kernel[grid](
        output_ptr = out,
        input_ptr = x,
        input_row_stride = x.stride(0), 
        output_row_stride = out.stride(0),
        n_rows = n_rows,
        n_cols = n_cols,
        BLOCK_SIZE = BLOCK_SIZE,
        num_stages = num_stages
    )
    
    return out


def test():
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=DEVICE)
    y_triton = softmax_triton(x)
    y_torch = torch.softmax(x, axis=1)
    assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
    print("PASS")
    return


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax', 'naive_softmax_compiled'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax", "Naive Softmax compiled"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('black', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax_triton(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    if provider == "naive_softmax_compiled":
        ms = triton.testing.do_bench(lambda: naive_softmax_compiled(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

benchmark_run = benchmark.run

# with A100-80GB

# softmax-performance:
#           N  Triton (GB/s)  Torch (GB/s)  Naive Softmax (GB/s)  Naive Softmax compiled (GB/s)
# ...
# 97  12672.0    1472.071583   1445.961393            390.482200                     983.504108