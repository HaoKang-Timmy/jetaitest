import tilelang
import tilelang.language as T
import sys  # noqa: F401
import itertools
from tilelang.autotuner import autotune
import time
import torch
# Add your fla repository path to sys.path
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae
# sys.path.insert(0, "/home/tzj/flash-linear-attention")




def get_configs():
    Block_T = [64 ,128, 256, 512]
    threads = [128, 256]
    _configs = list(itertools.product(Block_T, threads))
    configs = [{
        'Block_T': c[0],
        'threads': c[1]
    } for c in _configs]
    return configs
# tilelang.disable_cache()
@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    out_idx=[-1]
)
def tl_chunk_cumsum(
    B,
    Token,
    H,
    chunk_size = 64,
    input_dtype = "bfloat16",
    output_dtype = "float32",
    Block_T = 128,
    threads = 256,
):
    assert Block_T % chunk_size == 0, "Block_T must be divisible by chunk_size"
    chunk_num = Block_T // chunk_size
    fragment_shape = [chunk_num, chunk_size]
    @T.prim_func
    def kernel(
        InputG: T.Tensor(shape=(B, Token, H), dtype=input_dtype),
        OutputG: T.Tensor(shape=(B, Token, H), dtype=output_dtype)
    ):
        with T.Kernel(T.ceildiv(Token, Block_T), B * H, threads=threads) as (bt, bbh):
            bb, bh = bbh // H, bbh % H
            InputG_shared = T.alloc_shared((Block_T), dtype=input_dtype)
            InputG_fragment = T.alloc_fragment((Block_T), dtype=output_dtype)
            T.copy(InputG[bb, bt * Block_T:(bt + 1) * Block_T, bh], InputG_shared)

            T.copy(InputG_shared, InputG_fragment)
            InputG_fragment_viewed = T.view(InputG_fragment, fragment_shape)
            T.cumsum(InputG_fragment_viewed, dim=1)
            InputG_fragment_viewed_reverse = T.view(InputG_fragment_viewed, [Block_T])
            T.copy(InputG_fragment_viewed_reverse, InputG_shared)
            T.copy(InputG_shared, OutputG[bb, bt * Block_T:(bt + 1) * Block_T, bh])
    return kernel

@tilelang.jit(out_idx=[-1])
def tilelang_chunk_scaled_dot_kkt_fwd(
    # task config
    B,
    S,
    H,
    DK,
    chunk_size=64,
    input_dtype="bfloat16",
    output_dtype="bfloat16",
    accum_dtype="float32",
    # kernel config
    block_S=64,
    block_DK=64,
    threads=256,
    num_stages=0,
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
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype, scope="shared")
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            A_shared = T.alloc_shared((block_S, block_S), dtype=output_dtype)
            Beta_K_fragment = T.alloc_fragment((block_S, block_DK), dtype=input_dtype)
            A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            # Tensor used for gated:
            G_shared = T.alloc_shared((block_S,), dtype=accum_dtype, scope="shared")
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


            T.copy(A_fragment, A_shared)
            T.copy(A_shared, A[bb, bs * block_S:(bs + 1) * block_S, bh, :])

    return kernel

def chunk_cumsum(g, chunk_size = 64):
    B, Token, H = g.shape
    kernel = tl_chunk_cumsum(B, Token, H, chunk_size)
    output = kernel(g)
    return output


if __name__ == "__main__":
    # kernel = tilelang_chunk_scaled_matmul_fwd(1, 1024, 12, 96)
    kernel = tilelang_chunk_scaled_dot_kkt_fwd(1, 1024, 12, 96)
    k = torch.randn(1, 1024, 12, 96, dtype=torch.bfloat16).cuda()
    beta = torch.randn(1, 1024, 12, dtype=torch.bfloat16).cuda()
    g = torch.randn(1, 1024, 12, dtype=torch.float32).cuda()
    output = kernel(k, beta, g)
    print(output)
    # kernel1 = tilelang_chunk_scaled_matmul_fwd(1, 1024, 12, 96)
    # output1 = kernel1(k, beta, g)
    # print(output1)

    # g = torch.randn(1, 32768, 32, dtype=torch.bfloat16).cuda()
    # output = chunk_cumsum(g)
    # print(output)