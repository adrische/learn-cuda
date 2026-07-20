import torch
from torch import Tensor
from jaxtyping import Float

import triton
import triton.language as tl

from einops import rearrange


def weighted_sum(x: Float[Tensor, "... D"], 
                 weight: Float[Tensor, "D"]
                 ) -> Float[Tensor, "..."]:
    """Torch implementation of a weighted sum along the last axis."""
    return (weight * x).sum(axis=-1)


# tl.make_block_ptr() # not sure why it doesn't seem to have documentation available


# The weighted sum will be implemented as a for loop over the row-dimension
# Each iteration processes a number of elements in the row dimension,
# does the multiplication by weight and computes the sum.
# There is an accumulator saving these partial sums.
# Actually - the above process is executed for a number of rows simultaneously.
# The loop over the rows is hidden by the program id.

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,
    output_ptr,
    x_stride_row, x_stride_dim, # these are torch strides of x ?
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr
):
    row_tile_idx = tl.program_id(0) # blockIdx.x , the loop over threadIdx.x seems to be hidden by Triton
    
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D), # assignment says (NUM_ROWS, D,) (additional ,) why?
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), # this is where the loop over rows is hidden
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0)
    )
    
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,), # process a full row per "program"
        block_shape=(D_TILE_SIZE, ),
        order=(0, )
    )
    
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )
    
    # this holds the accumulation of the processed pieces of the rows
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    
    # loop over pieces of the row, each processing (up to) D_TILE_SIZE many elements (last one may be shorter)
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # load data
        row_tile_implicitly_for_many_rows = tl.load( # (ROWS_TILE_SIZE, D_TILE_SIZE)
            pointer=x_block_ptr, 
            mask=None,
            other=None,
            boundary_check=(0, 1), # dimensions which should do boundary checks
            padding_option="zero"
        )
        weigth_tile_same_for_all_rows = tl.load( # (D_TILE_SIZE,)
            pointer=weight_block_ptr,
            mask=None,
            other=None,
            boundary_check=(0, ),
            padding_option="zero"
        )
        
        # aggregrate  (ROWS_TILE_SIZE, D_TILE_SIZE) * (...,  D_TILE_SIZE) -> (ROWS_TILE_SIZE, )
        output += tl.sum(row_tile_implicitly_for_many_rows * weigth_tile_same_for_all_rows[None, :], axis=1)
        
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE, ))
        
    # after aggregation along the row (for several rows as processed in this program)
    tl.store(pointer=output_block_ptr, value=output, mask=None, boundary_check=(0,))


@triton.jit
def weighted_sum_bwd(
    x_ptr, weight_ptr,
    grad_output_ptr, # gradient to propagate
    grad_x_ptr, partial_grad_weight_ptr, # results, still need to do final reduction (sum) of patrial_grad_weight along row dim
    stride_xr, stride_xd,   # strides of x, weights, output gradients wrt x and partial output gradients with w
    stride_wd,              # this is nothing more than overhead? just passing this info from torch.stride?
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D, # full sizes, for guarding against accesses outside of the tensors
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr
):
    """Gradient propagation with respect do x and weights for weighted sum
    
    Gradient propagation with respect to x is outer product of gradient-to-propagate with weights
    Gradient propagation with respect to weights is x * gradient-to-propagate, but need to sum over rows
    as each row has the same weight.
    Sums are calculated in two steps: 
    1) sum per block calculatedwithin kernel, 
    2) sum of block-sums calculated in Pytorch
    """
    
    row_tile_idx = tl.program_id(0) # offset used along rows
    n_row_tiles = tl.num_programs(0) # gridDim.x ? how many partial sums remain, launch parameter
    
    # Block pointers, with strides, initial offsets for all in- and outputs
    grad_output_block_ptr = tl.make_block_ptr(
        base=grad_output_ptr,
        shape=(NUM_ROWS, ),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )
    
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0) # why is the order reversed?
    )
    
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )
    
    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0) # why not (0, 1)?
    )
    
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        base=partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0) # why reversed?
    )
    
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        row = tl.load(x_block_ptr, boundary_check=(0,1), padding_option="zero")
        
        # outer product for one block of size ROWS_TILE_SIZE * D_TILE_SIZE for grad_x
        # all possible combinations by block (kernel launch parameter) * for loop iteration
        grad_x_row = grad_output[:, None] * weight[None, :]
        
        # point-wise product of x * grad_output, grad_output is a row piece
        # the sum is taken along the row-dimension for as many rows as are processed here
        # this still leaves the sum un-finished - still need to accumulate gridDim many partial sums after kernel
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        
        # saving
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0,1))
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))
        
        # advance pointers (only along D dimension, 
        # the other dimension, if applicable, is "advanced" by the blocks (launch parameter))
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
        # grad_output_block_ptr only advanced along program dimension (blockIdx)
        
    

# This just seems to be the syntax
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor): # ctx = context? explicit argument for staticmethod, a torch.autograd.Function, I believe
        
        input_shape = x.shape
        D, output_dims = input_shape[-1], input_shape[:-1]
        
        # reshape to 2D
        x = rearrange(x, "... d -> (...) d")
        
        assert x.is_cuda and weight.is_cuda, "Expected Cuda tensors"
        assert len(weight.shape) == 1 and weight.shape[0] ==  D, "Dimension mismatch"
        assert x.is_contiguous(), "our pointer arithmetic assumes contiguous x expected"
        
        # Cache x, y for backward
        ctx.save_for_backward(x, weight)
        
        # Launch parameters - in the triton examples this is its own function
        # are the ctx.D_TILE_SIZE etc assignments used in backward?
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape
        
        y = torch.empty(output_dims, device=x.device)
        
        # Kernel launch
        n_rows = y.numel() # why not output_dims
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE), )](
            x, weight, # don't need to call to_pointer on these (or so?)
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE
        )
        
        return y.view(input_shape[:-1]) # input_shape[:-1] is just output_dims, which is anyway the shape of y?
    
    
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        
        n_rows, D = x.shape
        
        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)
        
        weighted_sum_bwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE
        )
        grad_weight = partial_grad_weight.sum(axis=0) # remaining reduction (sum) over number of blocks (launch parameter) many partial sums

        return grad_x, grad_weight

f_weightedsum = WeightedSumFunc.apply


def test():
    x1 = torch.rand((7, 10), device="cuda", requires_grad=True)
    w1 = torch.rand((10, ), device="cuda", requires_grad=True)

    x2, w2 = x1.detach().clone().requires_grad_(), w1.detach().clone().requires_grad_()


    out1 = weighted_sum(x1, w1)
    out2 = f_weightedsum(x2, w2)

    assert torch.allclose(out1, out2)

    loss1 = out1.mean()
    loss2 = out2.mean()

    loss1.backward()
    loss2.backward()

    assert torch.allclose(x1.grad, x2.grad)
    assert torch.allclose(w1.grad, w2.grad)
    
    print("Pass")