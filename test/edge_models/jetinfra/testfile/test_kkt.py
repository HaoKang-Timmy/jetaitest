import tilelang
import torch
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
from typing import Optional, Tuple
import time
import tilelang as tl
import sys
@tilelang.jit(out_idx=[-1])
def tilelang_chunk_scaled_dot_kkt_fwd(
    # task config
    B,
    S,
    H,
    DK,
    chunk_size=64,
    input_dtype="bfloat16",
    output_dtype="float32",
    accum_dtype="float32",
    # kernel config
    block_S=64,
    block_DK=64,
    threads=128,
    num_stages=2,
):
    K_shape = (B, S, H, DK)
    Beta_shape = (B, S, H)
    G_shape = (B, S, H)
    assert chunk_size == block_S, "chunk_size must be equal to block_S"
    BS = chunk_size
    output_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
            K: T.Tensor(K_shape, dtype=input_dtype),
            Beta: T.Tensor(Beta_shape, dtype=input_dtype),
            G: T.Tensor(G_shape, dtype=accum_dtype),
            A: T.Tensor(output_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            # !! Pay attention to the scope of the shared memory: may cause misaligned address when shape is one dimension or the buffer is too small
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            A_shared = T.alloc_shared((block_S, block_S), dtype=output_dtype)
            Beta_K_fragment = T.alloc_fragment((block_S, block_DK), dtype=input_dtype)
            A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            # Tensor used for gated:
            G_shared = T.alloc_shared((block_S,), dtype=accum_dtype)
            G_diff_local = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            T.annotate_layout({
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
            })

            # T.fill(A_fragment, 0)
            T.clear(A_fragment)
            T.disable_warp_group_reg_alloc()
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(
                    K[bb, bs * block_S:(bs + 1) * block_S, bh, i_k * block_DK:(i_k + 1) * block_DK],
                    K_shared)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    Beta_K_fragment[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s]
                T.gemm(Beta_K_fragment, K_shared, A_fragment, transpose_B=True)


            for i_s in T.Parallel(block_S):
                G_shared[i_s] = G[bb, bs * block_S + i_s, bh]
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                G_diff_local[i_s1, i_s2] = G_shared[i_s1] - G_shared[i_s2]
            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                with T.If(G_diff_local[i_s1, i_s2] <= 0 and i_s1 > i_s2):
                    with T.Then():
                        A_fragment[i_s1, i_s2] = A_fragment[i_s1, i_s2] * T.exp(
                            G_diff_local[i_s1, i_s2])
                    with T.Else():
                        A_fragment[i_s1, i_s2] = 0
            # for i_s1, i_s2 in T.Parallel(block_S, block_S):
            #     with T.If(G_frag[i_s1] <= G_frag[i_s2] and i_s1 > i_s2):
            #         with T.Then():
            #             A_fragment[i_s1, i_s2] = A_fragment[i_s1, i_s2] * T.exp(
            #                 G_frag[i_s1] - G_frag[i_s2])
            #         with T.Else():
            #             A_fragment[i_s1, i_s2] = 0

            T.copy(A_fragment, A_shared)
            T.copy(A_shared, A[bb, bs * block_S:(bs + 1) * block_S, bh, :])

    return kernel

def chunked_scaled_dot_ktt(k, beta, g_cumsum):
    B, Token, H, DK = k.shape
    kernel = tilelang_chunk_scaled_dot_kkt_fwd(B, Token, H, DK)
    output = kernel(k, beta, g_cumsum)
    torch.cuda.synchronize()
    print("over")
    return output

if __name__ == "__main__":
    B, S, H, DK = 1, 64, 12, 256
    datatype = torch.bfloat16
    acc_dtype = torch.float32
    eps = 1e-5
    autotune_interval = 524288
    
    # 准备输入数据
    K = torch.randn(B, S, H, DK, device='cuda', dtype=datatype)
    Beta = torch.randn(B, S, H, device='cuda', dtype=datatype)
    G_cumsum = torch.randn(B, S, H, device='cuda', dtype=acc_dtype)
    output = chunked_scaled_dot_ktt(K, Beta, G_cumsum)

    print(output)
