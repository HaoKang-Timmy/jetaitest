import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
import warnings
from typing import Optional
import itertools
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h

import torch
def check_tensors_and_compute_errors(t1, names=None):

    has_nan_1 = torch.isnan(t1).any().item()
    has_inf_1 = torch.isinf(t1).any().item()
    # print(f"{names} has_inf_1: {has_inf_1}")
    print(f"{names} has_nan_1: {has_nan_1}")


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    }
)
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
    threads=128,
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
            
            V_new: T.Tensor(V_shape, dtype=output_dtype),
            final_state: T.Tensor(final_state_shape, dtype=state_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            # b_h_shared = T.alloc_shared((DK, block_DV), dtype=input_dtype)
            # b_h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            b_h_shared = T.alloc_shared((block_DK, block_DV), dtype=output_dtype)
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
            # for i_s in T.serial(T.ceildiv(S, block_S)):
                # Store previous result to the hidden tensor, like the epilogue
                T.copy(b_h_shared[0:DK, 0:block_DV], h[bb, i_s, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])
                for i, j in T.Parallel(block_S, block_DK):

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
    h = k.new_empty(B, (Token + chunk_size - 1) // chunk_size, H, K, V)
    final_state = k.new_empty(B, H, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None
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
    kernel(k, w, u, g, h, v_new, final_state)
    
    return h, v_new, final_state


if __name__ == "__main__":
    B, Token, H, K, V = 1, 1024, 128, 128, 128
    input_dtype = torch.bfloat16
    element_dtype = torch.float32

    # k = torch.randn(B, Token, H, K).to(input_dtype).to("cuda")
    # w = torch.randn(B, Token, H, K).to(input_dtype).to("cuda")
    # u = torch.randn(B, Token, H, V).to(input_dtype).to("cuda")
    # g = torch.randn(B, Token, H).to(element_dtype).to("cuda")
    # k_copy = k.clone()
    # w_copy = w.clone()
    # u_copy = u.clone()
    # g_copy = g.clone()
    k = torch.load("./k.pt", map_location="cuda")
    w = torch.load("./w.pt", map_location="cuda")
    u = torch.load("./u.pt", map_location="cuda")
    g = torch.load("./g.pt", map_location="cuda")
    print("k shape: ", k.shape)
    print("w shape: ", w.shape)
    print("u shape: ", u.shape)
    print("g shape: ", g.shape)
    h, v_new, final_state = tilelang_chunk_gated_delta_rule(1, k, w, u, g)
    h = h.to(torch.bfloat16)
    v_new = v_new.to(torch.bfloat16)
    final_state = final_state.to(torch.bfloat16)
    check_tensors_and_compute_errors(h,"h")
    check_tensors_and_compute_errors(v_new,"v_new")
    check_tensors_and_compute_errors(final_state,"final_state")
    # h_triton, v_new_triton, final_state_triton = chunk_gated_delta_rule_fwd_h(k_copy, w_copy, u_copy, g_copy,
    # initial_state = None,
    # output_final_state = True,
    # cu_seqlens=None,
    # )
    
    # # 检查三个输出的误差
    # # h的误差
    # h_abs_rel_error = torch.mean(torch.abs(h - h_triton) / (torch.abs(h_triton) + 1e-8)).item()
    # h_mean_rel_error = torch.mean((h - h_triton) / (torch.abs(h_triton) + 1e-8)).item()
    # print(f"h - 平均绝对相对误差: {h_abs_rel_error:.6e}, 平均相对误差: {h_mean_rel_error:.6e}")
    
    # # v_new的误差
    # v_new_abs_rel_error = torch.mean(torch.abs(v_new - v_new_triton) / (torch.abs(v_new_triton) + 1e-8)).item()
    # v_new_mean_rel_error = torch.mean((v_new - v_new_triton) / (torch.abs(v_new_triton) + 1e-8)).item()
    # print(f"v_new - 平均绝对相对误差: {v_new_abs_rel_error:.6e}, 平均相对误差: {v_new_mean_rel_error:.6e}")
    
    # # final_state的误差
    # final_state_abs_rel_error = torch.mean(torch.abs(final_state - final_state_triton) / (torch.abs(final_state_triton) + 1e-8)).item()
    # final_state_mean_rel_error = torch.mean((final_state - final_state_triton) / (torch.abs(final_state_triton) + 1e-8)).item()
    # print(f"final_state - 平均绝对相对误差: {final_state_abs_rel_error:.6e}, 平均相对误差: {final_state_mean_rel_error:.6e}")
    