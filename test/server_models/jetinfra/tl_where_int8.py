import torch
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
import time
import torch.nn as nn


def get_configs():
    block_M = [32, 64, 128, 256]

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit()
def where_int8(
    Batch,
    SeqLen,
    dtype,
    output_dtype,
    block_M,
    threads,
):
    from tilelang.carver.arch import driver
    sm_num = driver.get_num_sms()
    @T.prim_func
    def main(
        Input: T.Tensor([Batch, SeqLen], dtype),
        Output: T.Tensor([Batch, SeqLen], output_dtype),
    ):
        with T.Kernel(Batch, T.ceildiv(SeqLen, block_M), threads = threads) as (bx, by):
            Input_shared = T.alloc_shared((block_M), dtype)
            input_reg = T.alloc_fragment((block_M), dtype)
            output_reg = T.alloc_fragment((block_M), output_dtype)
            output_shared = T.alloc_shared((block_M), output_dtype)
            T.copy(Input[bx, by * block_M], Input_shared)
            T.copy(Input_shared, input_reg)
            for i in T.Parallel(block_M):
                output_reg[i] = T.if_then_else()
            
            
            
            
    