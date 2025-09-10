import argparse
import itertools

import tilelang as tl
import tilelang.language as T
from tvm import DataType
from tilelang.autotuner import autotune
from tilelang import jit
import torch
import time


def ref_program(A, B):
    return A @ B.T


# @tl.jit(out_idx=[-1])
# def splitk_gemv_vectorized(
#     N: int,
#     K: int,
#     BLOCK_N: int,
#     reduce_threads: int,
#     dtype: str = "float16",
#     accum_dtype: str = "float",
# ):
#     MAX_TRANSACTION_SIZE_IN_BITS = 128
#     TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
#     BLOCK_K = reduce_threads * TILE_K

#     @T.prim_func
#     def main(
#             A: T.Tensor((K,), dtype),
#             B: T.Tensor((N, K), dtype),
#             C: T.Tensor((N,), dtype),
#     ):
#         with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
#             tn = T.get_thread_binding(0)
#             tk = T.get_thread_binding(1)
#             A_local = T.alloc_local((TILE_K,), dtype)
#             B_local = T.alloc_local((TILE_K,), dtype)
#             C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
#             C_accum = T.alloc_local((1,), accum_dtype)
#             if tk == 0:
#                 C_shared[tn] = 0
#             T.clear(C_accum)
#             for bk in T.serial(T.ceildiv(K, BLOCK_K)):
#                 for k in T.vectorized(TILE_K):
#                     A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
#                     B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
#                 for k in T.serial(TILE_K):
#                     C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
#             T.atomic_add(C_shared[tn], C_accum[0])
#             C[bn * BLOCK_N + tn] = C_shared[tn]

#     return main
def get_configs():
 
    BLOCK_N = [2, 4, 8, 32, 64, 128]  

    reduce_threads = [4, 8, 32]
    _configs = list(itertools.product(BLOCK_N, reduce_threads))
    configs = [{
        'BLOCK_N': c[0],
        'reduce_threads': c[1],
    } for c in _configs]
    return configs
@autotune(configs=get_configs(), warmup=10, rep=10)
@tl.jit(
    pass_configs={
        tl.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    }
)
def splitk_gemv_vectorized(
    Batch: int,
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
            A: T.Tensor((Batch, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((Batch, N), dtype),
    ):
        with T.Kernel(Batch, T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as (batch_id, bn):
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)

            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[batch_id, bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[batch_id, bn * BLOCK_N + tn] = C_shared[tn]

    return main


# def get_best_config(N, K):

#     def get_configs():
#         iter_params = dict(BLOCK_N=[2, 4, 8, 32, 64, 128], reduce_threads=[4, 8, 32])
#         return [
#             dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())
#         ]

#     @autotune(
#         configs=get_configs(),
#         warmup=3,
#         rep=20,
#     )
#     @jit(
#         out_idx=[-1],
#         target="auto",
#     )
#     def kernel(
#         BLOCK_N=None,
#         reduce_threads=None,
#     ):
#         dtype = "float16"
#         accum_dtype = "float"
#         MAX_TRANSACTION_SIZE_IN_BITS = 128
#         TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
#         BLOCK_K = reduce_threads * TILE_K

#         @T.prim_func
#         def main(
#                 A: T.Tensor((K,), dtype),
#                 B: T.Tensor((N, K), dtype),
#                 C: T.Tensor((N,), dtype),
#         ):
#             with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
#                 tn = T.get_thread_binding(0)
#                 tk = T.get_thread_binding(1)
#                 A_local = T.alloc_local((TILE_K,), dtype)
#                 B_local = T.alloc_local((TILE_K,), dtype)
#                 C_accum = T.alloc_local((1,), accum_dtype)

#                 T.clear(C_accum)
#                 for bk in T.serial(T.ceildiv(K, BLOCK_K)):
#                     for k in T.vectorized(TILE_K):
#                         A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
#                         B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
#                     for k in T.serial(TILE_K):
#                         C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(
#                             accum_dtype)
#                 C_reduced = T.alloc_local((1,), accum_dtype)
#                 with T.attr(
#                         T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
#                         "reduce_scope",
#                         T.reinterpret(T.uint64(0), dtype="handle"),
#                 ):
#                     T.evaluate(
#                         T.tvm_thread_allreduce(
#                             T.uint32(1),
#                             C_accum[0],
#                             True,
#                             C_reduced[0],
#                             tk,
#                             dtype="handle",
#                         ))

#                 C[bn * BLOCK_N + tn] = C_reduced[0]

#         return main

#     return kernel()


def check_correctness_and_bench(kernel, N, K, bench_ref=True):
    profiler = kernel.get_profiler()
    profiler.assert_allclose(lambda A, B, C: (A @ B.T,), atol=1e-2, rtol=1e-2)
    if bench_ref:
        latency = profiler.do_bench(lambda A, B, C: (A @ B.T,), warmup=50)
        print(f"Torch Latency: {latency} ms")
    latency = profiler.do_bench(kernel, warmup=50)
    print(f"TileLang Latency: {latency} ms\n")


def main():
    parser = argparse.ArgumentParser(description="GEMV Example")
    parser.add_argument("--n", type=int, default=1536, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=1152, help="Matrix dimension K")
    parser.add_argument("--batch", type=int, default=10, help="Batch dimension B")
    args, _ = parser.parse_known_args()
    N, K, Batch = args.n, args.k, args.batch
    # check_correctness_and_bench(naive_gemv(N, K, 128, 128), N, K)
    # check_correctness_and_bench(naive_splitk_gemv(N, K, 32, 32), N, K)
    # check_correctness_and_bench(splitk_gemv(N, K, 32, 32, 32), N, K)
    # check_correctness_and_bench(splitk_gemv_vectorized(N, K), N, K)
    # check_correctness_and_bench(splitk_gemv_vectorized_tvm(N, K, 2, 32), N, K)
    # print("Test passed!")

    # best_result = get_best_config(N, K)
    # best_config = best_result.config
    # kernel = splitk_gemv_vectorized_tvm(N, K, **best_config)
    # profiler = kernel.get_profiler()
    # latency = profiler.do_bench(lambda x, y: x @ y.T, warmup=500)
    # print(f"Torch Latency: {latency} ms")
    # latency = profiler.do_bench(kernel, warmup=500)
    # print(f"TileLang Latency: {latency} ms\n")
    kernel = splitk_gemv_vectorized(Batch, N, K)
    input_a = torch.randn(Batch, K).half().to("cuda")
    input_b = torch.randn(N, K).half().to("cuda")
    input_c = torch.randn(Batch, N).half().to("cuda")
    for _ in range(10):
        kernel(input_a, input_b, input_c)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(20):
        kernel(input_a, input_b, input_c)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Time: {end_time - start_time} s")
    print("Test passed!")



if __name__ == "__main__":
    main()