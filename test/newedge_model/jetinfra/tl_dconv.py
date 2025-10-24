import torch
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
import time
import torch.nn as nn
import numpy as np

dtype_dict = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "float8_e4m3",
    torch.float8_e5m2: "float8_e5m2",
}
def get_configs():
    block_D = [64 ,128, 256, 512]
    threads = [128]
    _configs = list(itertools.product(block_D, threads))
    configs = [{
        'block_D': c[0],
        'threads': c[1]
    } for c in _configs]
    return configs

# @autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx = [-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    }
)
def _dconv_fwd_kernel(
    Batch,
    Token,
    Kernel_size,
    Indim,
    block_D = 256,
    threads = 128,
    dtype = "bfloat16",  
    reduce_type = "float32"
):
    @T.prim_func
    def main_fp16_prefill(
        Input: T.Tensor([Batch * Token, Indim], dtype),
        Kernel_input: T.Tensor([Batch * Token, Kernel_size, Indim], dtype),
        Output: T.Tensor([Batch * Token, Indim], dtype),
        # Cache: T.Tensor([Batch * Token, T_cache], dtype),
    ):
        with T.Kernel(Batch * Token, T.ceildiv(Indim, block_D), threads=threads) as (bx, by):
            Kernel_shared = T.alloc_shared([Kernel_size, block_D], dtype, scope="shared")
            Input_shared = T.alloc_shared([Kernel_size, block_D], dtype, scope="shared")
            Output_shared = T.alloc_shared([block_D], dtype, scope="shared")
            Input_reg = T.alloc_fragment([Kernel_size, block_D], reduce_type)
            Output_reg = T.alloc_fragment([Kernel_size, block_D], reduce_type)
            Output_reduced = T.alloc_fragment([block_D], reduce_type)
            Kernel_reg = T.alloc_fragment([Kernel_size, block_D], reduce_type)
            batch_id = bx // Token
            token_id = bx % Token

            T.disable_warp_group_reg_alloc()
            # T.copy(Kernel_input[bx, by * block_D,0], Kernel_shared)
            T.copy(Kernel_input[bx, 0:Kernel_size, by * block_D:by * block_D + block_D], Kernel_shared)
            # for i, j in T.Parallel(block_D, Kernel_size):
            #     Kernel_shared[i, j] = Kernel_input[bx, i + by * block_D, j]
            ## copy input
            for i, j in T.Parallel(Kernel_size, block_D):
                Input_shared[i, j] = T.if_then_else(token_id + 1 + i - Kernel_size >= 0, 
                                                  Input[bx - Kernel_size + 1 + i, j + by * block_D], 0.0)
            
            T.copy(Input_shared, Input_reg)
            T.copy(Kernel_shared, Kernel_reg)
            for i, j in T.Parallel(Kernel_size, block_D):
                Output_reg[i, j] = Input_reg[i, j] * Kernel_reg[i, j]  
            T.reduce_sum(Output_reg, Output_reduced, dim=0)
            ### apply silu
            for i in T.Parallel(block_D):
                Output_reduced[i] = Output_reduced[i] / (1 + T.exp(-Output_reduced[i]))
            T.copy(Output_reduced, Output_shared)
            T.copy(Output_shared, Output[bx, by * block_D])
    return main_fp16_prefill
def tl_dynamic_conv_cache_w_silu(x, kernels):

    B, Token, D = x.shape
    W = kernels.shape[-2]
    dtype = dtype_dict[x.dtype]
    # out = torch.empty_like(x)
    #### TODO hard code to float16
    kernel = _dconv_fwd_kernel(B, Token, W, D, dtype = dtype)
    x = x.contiguous().view(B * Token, D)
    kernels = kernels.contiguous().view(B * Token, W, D)
    # out = out.view(B * Token, D)
    out = kernel(x, kernels)
    x = x.view(B, Token, D)
    out = out.view(B, Token, D)
    return out

if __name__ == "__main__":
    batch = 4
    token = 10
    indim = 512
    kernel_size = 4
    block_d = 32
    num_stages = 1
    threads = 128
    dtype = "float16"
    # _dconv_fwd_kernel(batch, token, indim, kernel_size, dtype = dtype)
    input1 = torch.randn(batch, token, indim).to(torch.float16).cuda()
    kernel_input = torch.randn(batch, token, kernel_size, indim).to(torch.float16).cuda()
    # output_triton = torch.randn(batch, token, indim).to(torch.float16).cuda()
    input1 = input1 * 5
    kernel_input = kernel_input * 5
    # kernel = _dconv_fwd_kernel(batch, token, indim, kernel_size, block_d, threads, dtype)
    # kernel(input1, kernel_input, output_triton)
    output_triton = tl_dynamic_conv_cache_w_silu(input1, kernel_input)
   