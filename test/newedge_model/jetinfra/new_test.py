
import tilelang
import tilelang.language as T
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
    use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
    # kernel config
    block_DK=64,
    block_DV=64,
    threads=256,
    num_stages=0,
):
    block_S = chunk_size
    BS = S // block_S

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    W_shape = (B, S, H, DK)
    U_shape = (B, S, H, DV)
    G_shape = (B, S, H)
    h_shape = (B, BS, H, DK, DV)
    initial_state_shape = (B, H, DK, DV)
    final_state_shape = (B, H, DK, DV)

    @T.prim_func
    def kernel(
            K: T.Tensor(K_shape, dtype=input_dtype),
            W: T.Tensor(W_shape, dtype=input_dtype),
            U: T.Tensor(U_shape, dtype=input_dtype),
            G: T.Tensor(G_shape, dtype=gate_dtype),
            initial_state: T.Tensor(initial_state_shape, dtype=input_dtype),
            h: T.Tensor(h_shape, dtype=output_dtype),
            final_state: T.Tensor(final_state_shape, dtype=state_dtype),
            V_new: T.Tensor(V_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            b_h_shared = T.alloc_shared((DK, block_DV), dtype=input_dtype)
            b_h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)

            U_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            V_new_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
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

            if use_initial_state:
                T.copy(initial_state[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV], b_h_shared)
                T.copy(b_h_shared, b_h_fragment)
            else:
                T.clear(b_h_fragment)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # Store previous result to the hidden tensor, like the epilogue
                T.copy(b_h_shared, h[bb, i_s, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

                # Recurrence
                T.copy(W[bb, i_s * block_S:(i_s + 1) * block_S, bh, 0:DK], W_shared)
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

                T.copy(K[bb, i_s * block_S:(i_s + 1) * block_S, bh, 0:DK], K_shared)
                # use_g
                if use_g:
                    G_last_local[0] = G[bb, (i_s + 1) * block_S - 1, bh]
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
                T.copy(b_h_fragment, final_state[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

    return kernel
def main():
    

if __name__ == "__main__":
    main()