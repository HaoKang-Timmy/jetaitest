import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
import warnings
from typing import Optional
import itertools
import torch

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True
    }
)
def tilelang_WUBV(
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
        V: T.Tensor(V_shape, dtype=input_dtype),
        W: T.Tensor(W_shape, dtype=input_dtype),
        U: T.Tensor(U_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        h: T.Tensor(h_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H