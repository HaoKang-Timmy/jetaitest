
import warnings
from typing import Optional

import torch
from einops import rearrange

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
import time

def chunk_gated_delta_rule_fwd(
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

    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    # obtain WY representation. u is actually the new v.
    start_time = time.time()

    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print("chunk_scaled_dot_kkt_fwd time:", end_time - start_time)
    start_time = time.time()
    # print("A shape:", A.shape)
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print("solve_tril time:", end_time - start_time)
    start_time = time.time()
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print("recompute_w_u_fwd time:", end_time - start_time)
    start_time = time.time()
    print("k shape:", k.shape)
    print("w shape:", w.shape)
    print("u shape:", u.shape)
    print("g shape:", g.shape)

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    print("h shape:", h.shape)
    print("v_new shape:", v_new.shape)
    print("final_state shape:", final_state.shape)
    torch.cuda.synchronize()
    end_time = time.time()
    print("chunk_gated_delta_rule_fwd_h time:", end_time - start_time)
    start_time = time.time()
    print("v_new shape:", v_new.shape)
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print("chunk_fwd_o time:", end_time - start_time)
    return g, o, A, final_state