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
    # out_idx = [-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    }
)
def _dconv_step_kernel(
    Batch,
    Token,
    Indim,
    Kernel_size,
    block_D,
    threads,
    dtype,
    reduce_type = "float32"
):
    @T.prim_func
    def main_fp16_prefill(
        Input: T.Tensor([Batch, Token, Indim], dtype),
        Cache: T.Tensor([Batch, Indim, Kernel_size], dtype),
        Kernel_input: T.Tensor([Batch, Token, Indim, Kernel_size], dtype),
        Output: T.Tensor([Batch, Token, Indim], dtype),
    ):
        with T.Kernel(Batch * Token, T.ceildiv(Indim, block_D), threads=threads) as (bx, by):
            Kernel_shared = T.alloc_shared([block_D, Kernel_size], dtype)
            Input_shared = T.alloc_shared([block_D, Kernel_size], dtype)
            Output_shared = T.alloc_shared([block_D], dtype)

            Input_reg = T.alloc_fragment([block_D, Kernel_size], reduce_type)
            Output_reg = T.alloc_fragment([block_D, Kernel_size], reduce_type)
            Output_reduced = T.alloc_fragment([block_D], reduce_type)
            Kernel_reg = T.alloc_fragment([block_D, Kernel_size], reduce_type)
            batch_id = bx // Token
            token_id = bx % Token
            for i, j in T.Parallel(block_D, Kernel_size):
                Kernel_shared[i, j] = Kernel_input[bx, token_id, i + by * block_D, j]
            for i, j in T.Parallel(block_D, Kernel_size):
                Input_shared[i, j] = T.if_then_else(token_id + 1 + j - Kernel_size >= 0, 
                                                  Input[bx, 1 + j - Kernel_size, i + by * block_D], Cache[bx, i + by * block_D, j+1])

            T.copy(Input_shared, Input_reg)
            T.copy(Kernel_shared, Kernel_reg)
            for i, j in T.Parallel(block_D, Kernel_size):
                Output_reg[i, j] = Input_reg[i, j] * Kernel_reg[i, j] 
            T.reduce_sum(Output_reg, Output_reduced, dim=1) 
            for i in T.Parallel(block_D):
                Output_reduced[i] = Output_reduced[i] / (1 + T.exp(-Output_reduced[i]))
            T.copy(Output_reduced, Output_shared)
            T.copy(Output_shared, Output[bx,token_id, by * block_D])
            ### renew cache
            T.copy(Input_shared, Cache[bx, by * block_D,0])
    return main_fp16_prefill

if __name__ == "__main__":
    batch = 4
    token = 1
    indim = 3072
    kernel_size = 4
    block_D = 128
    threads = 128
    dtype = "float16"

    input1 = torch.randn(batch, token, indim).to(torch.float16).cuda()
    kernel_input = torch.randn(batch, token, indim, kernel_size).to(torch.float16).cuda()
    cache = torch.randn(batch, indim, kernel_size).to(torch.float16).cuda()
    output_triton = torch.randn(batch, token, indim).to(torch.float16).cuda()
    kernel = _dconv_step_kernel(batch, token, indim, kernel_size, block_D, threads, dtype)
    kernel(input1, cache, kernel_input, output_triton)

            
            


