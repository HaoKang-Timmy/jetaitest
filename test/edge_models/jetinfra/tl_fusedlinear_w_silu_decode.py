import argparse
import itertools
import tilelang as tl
import tilelang
import tilelang.language as T
from tvm import DataType
from tilelang.autotuner import autotune
from tilelang import jit
import torch
import time

def get_configs():
    block_N = [64, 128]
    reduce_threads = [1, 2, 3, 4]
    _configs = list(itertools.product(block_N, reduce_threads))
    configs = [{

        'block_N': c[0],
        'reduce_threads': c[1],
    } for c in _configs]
    return configs

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    }
)
def splitk_gemv_vectorized(
    N: int,
    K: int,
    block_N: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
            A: T.Tensor((K,), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=(block_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((block_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * block_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * block_N + tn] = C_shared[tn]

    return main

if __name__ == "__main__":
    B, Token, D_in, D_out = 1, 1, 1536, 1152

    print("Benchmarking FP16...")
    # Tile-lang implementation
    Input_fp16 = torch.randn(B, Token, D_in).cuda().half()
    W_T_fp16 = torch.randn(D_out, D_in).cuda().half()
    Output_fp16 = torch.empty(B, Token, D_out).cuda().half()

    kernel_fp16 = splitk_gemv_vectorized(
        N=D_out, K=D_in, dtype="float16", accum_dtype="float"
    )

    input_vec = Input_fp16.view(D_in)
    output_vec = Output_fp16.view(D_out)

    # # warm up
    # for _ in range(10):
    #     kernel_fp16(input_vec, W_T_fp16, output_vec)

    # # benchmark
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(20):
    #     kernel_fp16(input_vec, W_T_fp16, output_vec)
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"Tilelang FP16 Time taken: {(end - start) * 1000 / 20} ms")

    # # PyTorch implementation
    # # warm up
    # for _ in range(10):
    #     output_ref = W_T_fp16 @ input_vec

    # # benchmark
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(20):
    #     output_ref = W_T_fp16 @ input_vec
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"Pytorch FP16 Time taken: {(end - start) * 1000 / 20} ms")

    # # Verify correctness
    # torch.testing.assert_close(output_vec, output_ref, rtol=1e-2, atol=1e-2)
    # print("Correctness check passed.")