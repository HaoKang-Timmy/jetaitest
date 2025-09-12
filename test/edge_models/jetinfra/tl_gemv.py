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


def get_configs():
 
    BLOCK_N = [128]   # Notice: Hard code to 128 cause need to fuse qknorm with gemv

    reduce_threads = [4, 8, 32]
    _configs = list(itertools.product(BLOCK_N, reduce_threads))
    configs = [{
        'BLOCK_N': c[0],
        'reduce_threads': c[1],
    } for c in _configs]
    return configs
# @autotune(configs=get_configs(), warmup=10, rep=10)
@tl.jit(
    pass_configs={
        tl.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    }
)
def splitk_gemv_vectorized_silu_l2norm(
    Batch: int,
    N: int,
    K: int,
    norm_dim: int,
    BLOCK_N: int = 128,
    reduce_threads: int = 4,
    dtype: str = "bfloat16",
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
        with T.Kernel(Batch, T.ceildiv(N, BLOCK_N), threads=(BLOCK_N * reduce_threads)) as (batch_id, bn):
            
            # tn = T.get_thread_binding(0)
            # tk = T.get_thread_binding(1)
            tx = T.get_thread_binding(0)
            tn = tx // reduce_threads
            tk = tx % reduce_threads
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            C_squared = T.alloc_fragment((BLOCK_N,), accum_dtype)
            sum_reg = T.alloc_fragment([1], accum_dtype)

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
            #### silu
            for i in T.Parallel(BLOCK_N):
                C_shared[i] = C_shared[i] / (1 + T.exp(-C_shared[i]))
            #### k2 norm
            for i in T.Parallel(BLOCK_N):
                C_squared[i] = C_shared[i] * C_shared[i]
            
            # if tn % norm_dim == 0:
            # L2Norm_QK(C_shared)
            T.reduce_sum(C_squared, sum_reg, dim=0)
            # for i in T.Parallel(1):
            for i in T.Parallel(1):
                sum_reg[i] = T.sqrt(sum_reg[i]) + 1e-6
            for i in T.Parallel(BLOCK_N):
                C_shared[i] = C_shared[i] / sum_reg[0]
            C[batch_id, bn * BLOCK_N + tn] = C_shared[tn]
    # print(main)
    return main
def gemv_silu_l2norm_kernel(
    input,
    weight_t,
):
    B, Token, K = input.shape
    N, _ = weight_t.shape
    torch_dtype = input.dtype
    device = input.device
    output = torch.empty(B, Token, N).to(torch_dtype).to(device)
    new_B = B * Token
    kernel = splitk_gemv_vectorized_silu_l2norm(new_B, N, K, norm_dim=128)
    kernel(input.view(new_B, K), weight_t, output.view(new_B, N))
    output = output.view(B, Token, N)
    return output




def check_correctness_and_bench(kernel, N, K, bench_ref=True):
    profiler = kernel.get_profiler()
    profiler.assert_allclose(lambda A, B, C: (A @ B.T,), atol=1e-2, rtol=1e-2)
    if bench_ref:
        latency = profiler.do_bench(lambda A, B, C: (A @ B.T,), warmup=50)
        print(f"Torch Latency: {latency} ms")
    latency = profiler.do_bench(kernel, warmup=50)
    print(f"TileLang Latency: {latency} ms\n")

def compare_pytorch(a, b, norm_dim=128):
    # Step 1: GEMV + SiLU
    output = (a @ b.T) / (1 + torch.exp(-(a @ b.T)))
    # output = (a @ b.T)

    # Step 2: Reshape for normalization
    reshaped_output = output.view(a.shape[0], -1, norm_dim)

    # Step 3: L2 Norm calculation
    # Keepdim=True to allow broadcasting for division
    norms = reshaped_output * reshaped_output
    norms = torch.sum(norms, dim=-1)
    norms = torch.sqrt(norms) + 1e-6
    
    # Step 4: Normalize
    # Add epsilon for numerical stability, same as in the kernel
    normalized_output = reshaped_output / (norms.unsqueeze(-1))

    # Step 5: Reshape back to original
    final_output = normalized_output.view(a.shape[0], -1)
    # final_output = output
    return final_output

def main():
    parser = argparse.ArgumentParser(description="GEMV Example")
    parser.add_argument("--n", type=int, default=128, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=1536, help="Matrix dimension K")
    parser.add_argument("--token", type=int, default=1, help="Token dimension T")
    parser.add_argument("--batch", type=int, default=1, help="Batch dimension B")
    args, _ = parser.parse_known_args()
    N, K, Batch, Token = args.n, args.k, args.batch, args.token

    kernel = splitk_gemv_vectorized_silu_l2norm(Batch * Token, N, K, norm_dim=128)

    # Manual Correctness Check
    print("--- Running Correctness Check ---")
    input_a = torch.randn(Batch, Token, K).to(torch.bfloat16).to("cuda")
    input_b = torch.randn(N, K).to(torch.bfloat16).to("cuda")
    output_c_tl = torch.empty(Batch, Token, N).to(torch.bfloat16).to("cuda")

    # Run TileLang Kernel
    kernel(input_a.view(Batch * Token, K), input_b, output_c_tl.view(Batch * Token, N))

    # Run PyTorch reference
    output_c_torch = compare_pytorch(input_a.view(Batch * Token, K), input_b, norm_dim=128)

    # Compare results
    abs_diff = (output_c_tl - output_c_torch).abs()
    rel_diff = abs_diff / (output_c_torch.abs() + 1e-8)

    print(f"Max absolute difference: {abs_diff.max().item():.6e}")
    print(f"Max relative error: {rel_diff.max().item():.6e}")
    print(f"Mean relative error: {rel_diff.mean().item():.6e}")

    # are_close = torch.allclose(output_c_tl, output_c_torch, atol=1e-4, rtol=1e-2)
    # print(f"Correctness check passed (torch.allclose): {are_close}")
    # if not are_close:
    #     print("Max difference (from torch.allclose):", (output_c_tl - output_c_torch).abs().max().item())

    # Performance Benchmarking
    print("\n--- Running Benchmarking ---")
    # Use new tensors for benchmark to avoid caching effects
    bench_a = torch.randn(Batch, Token, K).to(torch.bfloat16).to("cuda")
    bench_b = torch.randn(N, K).to(torch.bfloat16).to("cuda")
    # bench_c = torch.empty(Batch, N).half().to("cuda")
    
    # Warmup
    for _ in range(10):
        gemv_silu_l2norm_kernel(bench_a, bench_b)
    torch.cuda.synchronize()
    
    # Timed run
    start_time = time.time()
    for _ in range(20):
        gemv_silu_l2norm_kernel(bench_a, bench_b)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Average TileLang Latency: {(end_time - start_time) / 20 * 1000:.4f} ms")


if __name__ == "__main__":
    main()