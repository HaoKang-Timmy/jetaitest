
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
import warnings
from typing import Optional
import itertools

import torch
from einops import rearrange

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from fla.modules.l2norm import l2norm_fwd
import time

from .tl_trilcompute import tl_solve_tril
@tilelang.jit(
    out_idx=[-1]
)
def tl_chunk_cumsum(
    B,
    Token,
    H,
    chunk_size = 64,
    input_dtype = "float32",
    output_dtype = "float32",
    Block_T = 128,
    threads = 128,
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
            InputG_shared = T.alloc_shared((Block_T), dtype=input_dtype, scope="shared")
            InputG_fragment = T.alloc_fragment((Block_T), dtype=output_dtype)
            # unable to use TMA
            # T.copy(InputG[bb, bt * Block_T:(bt + 1) * Block_T, bh], InputG_shared)
            for i in T.Parallel(Block_T):
                InputG_shared[i] = InputG[bb, bt * Block_T + i, bh]
            T.copy(InputG_shared, InputG_fragment)
            InputG_fragment_viewed = T.view(InputG_fragment, fragment_shape)
            T.cumsum(InputG_fragment_viewed, dim=1)
            InputG_fragment_viewed_reverse = T.view(InputG_fragment_viewed, [Block_T])
            T.copy(InputG_fragment_viewed_reverse, InputG_shared)
            # unable to use TMA
            # T.copy(InputG_shared, OutputG[bb, bt * Block_T:(bt + 1) * Block_T, bh])
            for i in T.Parallel(Block_T):
                OutputG[bb, bt * Block_T + i, bh] = InputG_shared[i]
    return kernel

def chunk_cumsum(g, chunk_size = 64):
    B, Token, H = g.shape
    kernel = tl_chunk_cumsum(B, Token, H, chunk_size)
    output = kernel(g)
    return output

def get_configs():
    block_S = [64]
    block_DK = [64, 96, 128]
    threads = [128, 256]
    num_stages = [1, 2, 3, 4]
    _configs = list(itertools.product(block_S, block_DK, threads, num_stages))
    configs = [{
        'block_S': c[0],
        'block_DK': c[1],
        'threads': c[2],
        'num_stages': c[3]
    } for c in _configs]
    return configs
# tilelang.disable_cache()
# @autotune(configs=get_configs(), warmup=10, rep=10)
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
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype, scope="shared")
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
    return output

def get_configs():
    block_DK = [64, 96, 128]
    block_DV = [64, 128, 256]
    threads = [128, 256]
    num_stages = [1, 2, 3]
    _configs = list(itertools.product(block_DK, block_DV, threads, num_stages))
    configs = [{
        'block_DK': c[0],
        'block_DV': c[1],
        'threads': c[2],
        'num_stages': c[3]
    } for c in _configs]
    return configs
@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[-2, -1])
def tilelang_recompute_w_u_fwd(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    gate_dtype,
    accum_dtype,
    chunk_size,
    # kernel config
    block_S=64,
    block_DK=64,
    block_DV=64,
    threads=256,
    num_stages=0,
):
    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    Beta_shape = (B, S, H)
    assert chunk_size == block_S, "chunk_size must be equal to block_S"
    BS = chunk_size
    G_shape = (B, S, H)
    A_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
            K: T.Tensor(K_shape, dtype=input_dtype),
            V: T.Tensor(V_shape, dtype=input_dtype),
            Beta: T.Tensor(Beta_shape, dtype=input_dtype),
            G: T.Tensor(G_shape, dtype=gate_dtype),
            A: T.Tensor(A_shape, dtype=output_dtype),
            W: T.Tensor(K_shape, dtype=output_dtype),
            U: T.Tensor(V_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype )
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            G_shared = T.alloc_shared((block_S,), dtype=gate_dtype )
            A_shared = T.alloc_shared((block_S, block_S), dtype=output_dtype)
            W_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            W_shared = T.alloc_shared((block_S, block_DK), dtype=output_dtype)
            U_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            W_Beta_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            U_Beta_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)

            T.annotate_layout({
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                W_Beta_shared: tilelang.layout.make_swizzled_layout(W_Beta_shared),
                U_Beta_shared: tilelang.layout.make_swizzled_layout(U_Beta_shared),
            })

            T.disable_warp_group_reg_alloc()
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]
                G_shared[i_s] = T.exp(G[bb, bs * block_S + i_s, bh])

            T.copy(A[bb, bs * block_S:(bs + 1) * block_S, bh, :], A_shared)

            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(
                    V[bb, bs * block_S:(bs + 1) * block_S, bh, i_v * block_DV:(i_v + 1) * block_DV],
                    V_shared)
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    U_Beta_shared[i_s, i_v2] = V_shared[i_s, i_v2] * Beta_shared[i_s]
                T.gemm(A_shared, U_Beta_shared, U_fragment, clear_accum=True)
                # First copy to smem, then copy to gmem to reduce U2RU instructions
                T.copy(U_fragment, U_shared)
                T.copy(
                    U_shared, U[bb, bs * block_S:(bs + 1) * block_S, bh,
                                i_v * block_DV:(i_v + 1) * block_DV])

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(
                    K[bb, bs * block_S:(bs + 1) * block_S, bh, i_k * block_DK:(i_k + 1) * block_DK],
                    K_shared)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    W_Beta_shared[i_s,
                                  i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s] * G_shared[i_s]
                T.gemm(A_shared, W_Beta_shared, W_fragment, clear_accum=True)
                # First copy to smem, then copy to gmem to reduce U2RU instructions
                T.copy(W_fragment, W_shared)
                T.copy(
                    W_shared, W[bb, bs * block_S:(bs + 1) * block_S, bh,
                                i_k * block_DK:(i_k + 1) * block_DK])

    return kernel

def  tl_recompute_wu_forward(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
):
    B, Token, H, K, V = *k.shape, v.shape[-1]
    kernel = tilelang_recompute_w_u_fwd(B, Token, H, K, V,chunk_size=64,
    input_dtype = "bfloat16",
    output_dtype = "bfloat16",
    gate_dtype = "float32",
    accum_dtype = "float32",
    )
    w, u  = kernel(k, v, beta, g_cumsum, A)
    return w, u

@tilelang.jit(out_idx=[-3, -2, -1])
def tilelang_chunk_gated_delta_rule_fwd_h(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    use_g=True,
    # use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
    # kernel config
    block_DK=128,
    block_DV=32,
    threads=256,
    num_stages=2,
):
    block_S = chunk_size
    BS = (S + block_S - 1) // block_S

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    W_shape = (B, S, H, DK)
    U_shape = (B, S, H, DV)
    G_shape = (B, S, H)
    h_shape = (B, BS, H, DK, DV)
    initial_state_shape = (B, H, DK, DV)
    final_state_shape = (B, H, DK, DV)
    assert block_DK >= DK
    @T.prim_func
    def kernel(
            K: T.Tensor(K_shape, dtype=input_dtype),
            W: T.Tensor(W_shape, dtype=input_dtype),
            U: T.Tensor(U_shape, dtype=input_dtype),
            G: T.Tensor(G_shape, dtype=gate_dtype),
            # initial_state: T.Tensor(initial_state_shape, dtype=input_dtype),
            h: T.Tensor(h_shape, dtype=output_dtype),
            final_state: T.Tensor(final_state_shape, dtype=state_dtype),
            V_new: T.Tensor(V_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            # b_h_shared = T.alloc_shared((DK, block_DV), dtype=input_dtype)
            # b_h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            b_h_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            b_h_fragment = T.alloc_fragment((block_DK, block_DV), dtype=accum_dtype)

            U_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            # W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            W_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_new_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            # K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            G_last_local = T.alloc_local((1), dtype=gate_dtype)
            G_shared = T.alloc_shared((block_S, block_DV), dtype=gate_dtype)
            G_fragment = T.alloc_fragment((block_S, block_DV), dtype=gate_dtype)

            T.annotate_layout({
                b_h_shared: tilelang.layout.make_swizzled_layout(b_h_shared),
                U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                W_shared: tilelang.layout.make_swizzled_layout(W_shared),
                V_new_shared: tilelang.layout.make_swizzled_layout(V_new_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                G_shared: tilelang.layout.make_swizzled_layout(G_shared),
            })

            T.use_swizzle(10)
            T.disable_warp_group_reg_alloc()
            # if use_initial_state:
            #     T.copy(initial_state[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV], b_h_shared)
            #     T.copy(b_h_shared, b_h_fragment)
            # else:
            T.clear(b_h_fragment)
            T.copy(b_h_fragment, b_h_shared)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # Store previous result to the hidden tensor, like the epilogue
                T.copy(b_h_shared[0:DK, 0:block_DV], h[bb, i_s, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])
                # for i, j in T.Parallel(DK, block_DV):
                #     h[bb, i_s, bh, i, bv * block_DV + j] = b_h_shared[i, j]

                # Recurrence
                # T.copy(W[bb, i_s * block_S:(i_s + 1) * block_S, bh, 0:DK], W_shared)
                for i, j in T.Parallel(block_S, block_DK):
                    # with T.If(i_s * block_S + i < S and j < block_DK):
                    #     with T.Then():
                    #         W_shared[i, j] = W[bb, i_s * block_S + i, bh, j]
                    #     with T.Else():
                    #         W_shared[i, j] = 0
                    W_shared[i, j] = W[bb, i_s * block_S + i, bh, j]
                
                T.gemm(W_shared, b_h_shared, V_new_fragment, clear_accum=True)
                # U - W * S
                T.copy(
                    U[bb, i_s * block_S:(i_s + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV],
                    U_shared)
                T.copy(U_shared, U_fragment)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    V_new_fragment[i_s2, i_v] = -V_new_fragment[i_s2, i_v] + U_fragment[i_s2, i_v]

                # Save V_new
                if save_new_value:
                    T.copy(V_new_fragment, dst=V_new_shared)
                    T.copy(
                        V_new_shared, V_new[bb, i_s * block_S:(i_s + 1) * block_S, bh,
                                            bv * block_DV:(bv + 1) * block_DV])

                # T.copy(K[bb, i_s * block_S:(i_s + 1) * block_S, bh, 0:DK], K_shared[0:block_S, 0:DK])
                for i, j in T.Parallel(block_S, DK):
                    K_shared[i, j] = K[bb, i_s * block_S + i, bh, j]
                # use_g
                if use_g:
                    last_idx = T.min((i_s + 1) * block_S, S) - 1
                    G_last_local[0] = G[bb, last_idx, bh]
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        G_shared[i_s2, i_v] = G[bb, i_s * block_S + i_s2, bh]
                    T.copy(G_shared, G_fragment)
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        with T.If(G_last_local[0] - G_fragment[i_s2, i_v] <= 0):
                            with T.Then():
                                V_new_fragment[i_s2, i_v] = V_new_fragment[i_s2, i_v] * T.exp(
                                    G_last_local[0] - G_fragment[i_s2, i_v])
                            with T.Else():
                                V_new_fragment[i_s2, i_v] = 0
                    G_last_local[0] = T.exp(G_last_local[0])
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_h_fragment[i_k, i_v] *= G_last_local[0]

                # Update intermediate results
                T.copy(V_new_fragment, V_new_shared)
                T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)

                T.copy(b_h_fragment, b_h_shared)

            # Save final state
            if store_final_state:
                T.copy(b_h_fragment[0:DK, 0:block_DV], final_state[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])
                # for i, j in T.Parallel(DK, block_DV):
                #     b_h_fragment[i, j] = final_state[bb, bh, i, bv * block_DV + j]

    return kernel

def tilelang_chunk_gated_delta_rule(
    batch_size: int,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,

):
    B, Token, H, K, V = *k.shape, u.shape[-1]

    kernel = tilelang_chunk_gated_delta_rule_fwd_h(B, Token, H, K, V,
        input_dtype = "bfloat16",
        output_dtype = "bfloat16",
        accum_dtype = "float32",
        gate_dtype = "float32",
        state_dtype = "float32",
        chunk_size = 64,
        use_g = g is not None,
        # use_initial_state = initial_state is not None,
        store_final_state = output_final_state,
        save_new_value = save_new_value,
    )
    h, final_state, v_new, = kernel(k, w, u, g)
    
    return h, v_new, final_state

def get_configs():
    block_DK = [32, 64, 128]
    block_DV = [32, 64, 128]
    threads = [128, 256]
    num_stages = [1, 2, 3, 4]
    _configs = list(itertools.product(block_DK, block_DV, threads, num_stages))
    configs = [{
        'block_DK': c[0],
        'block_DV': c[1],
        'threads': c[2],
        'num_stages': c[3]
    } for c in _configs]
    return configs
@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(out_idx=[-1],
pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    })
def tilelang_chunk_fwd_o(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    scale,
    use_g,
    # kernel config
    block_S=64,
    block_DK=64,
    block_DV=64,
    threads=128,
    num_stages=1,
):
    assert chunk_size == block_S, "chunk_size must be equal to block_S"
    BS = chunk_size
    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    H_shape = (B, (S + block_S - 1) // BS, H, DK, DV)
    G_shape = (B, S, H)
    O_shape = (B, S, H, DV)

    @T.prim_func
    def kernel(
            Q: T.Tensor(Q_shape, dtype=input_dtype),
            K: T.Tensor(K_shape, dtype=input_dtype),
            V: T.Tensor(V_shape, dtype=input_dtype),
            HIDDEN: T.Tensor(H_shape, dtype=input_dtype),
            G: T.Tensor(G_shape, dtype=gate_dtype),
            O: T.Tensor(O_shape, dtype=output_dtype),
    ):
        with T.Kernel(
                T.ceildiv(DV, block_DV), T.ceildiv(S, block_S), B * H,
                threads=threads) as (bv, bs, bbh):
            bb, bh = bbh // H, bbh % H
            Q_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            H_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            O_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            G_shared = T.alloc_shared((block_S,), dtype=gate_dtype )
            G_diff_local = T.alloc_fragment((block_S, block_S), dtype=gate_dtype)

            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                H_shared: tilelang.layout.make_swizzled_layout(H_shared),
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                O_shared: tilelang.layout.make_swizzled_layout(O_shared),
            })

            T.clear(A_fragment)
            T.clear(O_fragment)
            T.disable_warp_group_reg_alloc()
            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(
                    Q[bb, bs * block_S:(bs + 1) * block_S, bh, i_k * block_DK:(i_k + 1) * block_DK],
                    Q_shared)
                T.copy(
                    K[bb, bs * block_S:(bs + 1) * block_S, bh, i_k * block_DK:(i_k + 1) * block_DK],
                    K_shared)
                T.copy(
                    HIDDEN[bb, bs, bh, i_k * block_DK:(i_k + 1) * block_DK,
                           bv * block_DV:(bv + 1) * block_DV], H_shared)
                T.gemm(Q_shared, H_shared, O_fragment)
                T.gemm(Q_shared, K_shared, A_fragment, transpose_B=True)

            if use_g:
                for i_s in T.Parallel(block_S):
                    G_shared[i_s] = G[bb, bs * block_S + i_s, bh]
                # T.copy(G[bb, bs * block_S:(bs + 1) * block_S, bh], G_shared)
                for i_s, i_v in T.Parallel(block_S, block_DV):
                    O_fragment[i_s, i_v] = O_fragment[i_s, i_v] * T.exp(G_shared[i_s])
                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    G_diff_local[i_s1, i_s2] = G_shared[i_s1] - G_shared[i_s2]
                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    with T.If(G_diff_local[i_s1, i_s2] <= 0):
                        with T.Then():
                            A_fragment[i_s1, i_s2] = A_fragment[i_s1, i_s2] * T.exp(
                                G_diff_local[i_s1, i_s2])
                        with T.Else():
                            A_fragment[i_s1, i_s2] = 0

            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                with T.If(i_s1 < i_s2):  # noqa: SIM117
                    with T.Then():
                        A_fragment[i_s1, i_s2] = 0

            T.copy(V[bb, bs * block_S:(bs + 1) * block_S, bh, bv * block_DV:(bv + 1) * block_DV],
                   V_shared)
            T.copy(A_fragment, A_shared)
            T.gemm(A_shared, V_shared, O_fragment)

            for i_s, i_v in T.Parallel(block_S, block_DV):
                O_fragment[i_s, i_v] = O_fragment[i_s, i_v] * scale

            T.copy(O_fragment, O_shared)
            T.copy(O_shared, O[bb, bs * block_S:(bs + 1) * block_S, bh,
                               bv * block_DV:(bv + 1) * block_DV])

    return kernel

def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,  # cumsum of log decay
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
):
    B, Token, H, K, V = *q.shape, v.shape[-1]
    kernel = tilelang_chunk_fwd_o(
        B, Token, H, K, V,
        input_dtype = "bfloat16",
        output_dtype = "bfloat16",
        accum_dtype = "float32",
        gate_dtype = "float32",
        chunk_size = 64,
        scale = scale,
        use_g = g is not None,

    )

    output = kernel(q, k, v, h, g)
    return output

def chunk_gated_delta_rule_fwd(
    batch_size: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None
):
    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    q = q.reshape(batch_size, -1, q.shape[-2], q.shape[-1])
    k = k.reshape(batch_size, -1, k.shape[-2], k.shape[-1])
    v = v.reshape(batch_size, -1, v.shape[-2], v.shape[-1])
    g = g.reshape(batch_size, -1, g.shape[-1])
    beta = beta.reshape(batch_size, -1, beta.shape[-1])
    # initial_state = initial_state.reshape(batch_size, -1, initial_state.shape[-2], initial_state.shape[-1])

    # g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    g = chunk_cumsum(g, chunk_size=64)
    # obtain WY representation. u is actually the new v.

    # A = chunk_scaled_dot_kkt_fwd(
    #     k=k,
    #     beta=beta,
    #     g_cumsum=g,
    #     cu_seqlens=cu_seqlens,
    #     output_dtype=torch.float32
    # )
    A = chunked_scaled_dot_ktt(k, beta, g)
    # torch.cuda.synchronize()
    # end_time = time.time()
    # print("chunk_scaled_dot_kkt_fwd time:", end_time - start_time)
    # start_time = time.time()
    
    A = A.reshape(1, -1, A.shape[-2], A.shape[-1])
    # print("A shape:", A.shape)
    # A = solve_tril(
    #     A=A,
    #     cu_seqlens=cu_seqlens,
    #     output_dtype=k.dtype
    # )
    A = tl_solve_tril(
        A=A,
    )
    A = A.reshape(batch_size, -1, A.shape[-2], A.shape[-1])
    # torch.cuda.synchronize()
    # end_time = time.time()
    # print("solve_tril time:", end_time - start_time)
    # start_time = time.time()
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    # w, u = tl_recompute_wu_forward(k, v, beta, g, A)
    # torch.cuda.synchronize()
    # end_time = time.time()
    # print("recompute_w_u_fwd time:", end_time - start_time)
    # start_time = time.time()
    # print("k shape:", k.shape)
    # print("w shape:", w.shape)
    # print("u shape:", u.shape)
    # print("g shape:", g.shape)

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    # h,  v_new, final_state = tilelang_chunk_gated_delta_rule(batch_size, k, w, u, g, output_final_state, chunk_size=64, save_new_value=True)
    # print("h shape:", h.shape)
    # print("v_new shape:", v_new.shape)
    # print("final_state shape:", final_state.shape)
    # torch.cuda.synchronize()
    # end_time = time.time()
    # print("chunk_gated_delta_rule_fwd_h time:", end_time - start_time)
    # start_time = time.time()
    # print("v_new shape:", v_new.shape)
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    # o = chunk_fwd_o(q, k, v_new, h, g, scale)
    # torch.cuda.synchronize()
    # end_time = time.time()
    # print("chunk_fwd_o time:", end_time - start_time)
    o = o.reshape(1, -1, o.shape[-2], o.shape[-1])
    return g, o, A, final_state