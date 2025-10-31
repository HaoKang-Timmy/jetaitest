import tilelang
import torch
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
from typing import Optional, Tuple
import time
import tilelang as tl
import sys
sys.path.insert(0, '/storage/home/hcoda1/6/hkang342/p-tkrishna3-0/jetaitest/flash-linear-attention')
from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule


def get_configs():
    block_K = [128]
    block_V = [128]
    num_stages = [3, 4]
    threads = [128]
    # head_split_k = [1, 2, 3, 4]
    # head_split_v = [1, 2, 3, 4]
    _configs = list(itertools.product(block_K, block_V, num_stages, threads))
    configs = [{
        'block_K': c[0],
        'block_V': c[1],
        'num_stages': c[2],
        'threads': c[3],
        # 'head_split_k': c[4],
        # 'head_split_v': c[5]
    } for c in _configs]
    return configs
#### TODO Fp8 loading, load multiple heads to enlarge block size, tensore core process instead of cuda core. We need to put norm in matmul.
# @autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_recurrent(
    Batch, 
    Token, 
    Head, 
    Head_V, 
    K_Dim, 
    V_Dim, 
    USE_QK_L2NORM_IN_KERNEL=True, 
    STORE_FINAL_STATE=True,
    dtype = "bfloat16",
    accum_dtype = "float32",
    block_K=128, 
    block_V=64, 
    num_stages=2, 
    threads=128
):
    @T.macro
    def L2Norm_QK(
        QK: T.SharedBuffer([block_K],accum_dtype),
    ):
        shared_reg = T.alloc_fragment([block_K], accum_dtype)
        squared_reg = T.alloc_fragment([block_K], accum_dtype)
        sum_reg = T.alloc_fragment([1], accum_dtype)
        T.copy(QK, shared_reg)
        
        # 计算元素的平方用于求norm
        for i in T.Parallel(block_K):
            squared_reg[i] = shared_reg[i] * shared_reg[i]
        T.reduce_sum(squared_reg, sum_reg, dim=0)
        sum_reg[0] = T.sqrt(sum_reg[0]) + 1e-6
        
        # 用原始元素除以norm
        for i in T.Parallel(block_K):
            shared_reg[i] = shared_reg[i] / sum_reg[0]
        T.copy(shared_reg, QK)

    @T.prim_func
    def main(
        Q: T.Tensor([Batch, Token, Head, K_Dim], dtype),
        K: T.Tensor([Batch, Token, Head, K_Dim], dtype),
        V: T.Tensor([Batch, Token, Head_V, V_Dim], dtype),
        g: T.Tensor([Batch, Token, Head_V], accum_dtype),
        Beta: T.Tensor([Batch, Token, Head_V],dtype),
        O: T.Tensor([Batch, Token, Head_V, V_Dim], dtype),
        h0: T.Tensor([Batch, Head_V, K_Dim, V_Dim], accum_dtype),
        # ht: T.Tensor([Batch, Head_V, K_Dim, V_Dim], dtype),
        scale: T.Tensor([1], dtype),
        # O: T.Tensor(Batch, Token, Head_V, V_Dim, dtype),
    ):
        with T.Kernel(T.ceildiv(K_Dim, block_K), T.ceildiv(V_Dim, block_V), T.ceildiv(Batch*Head_V, 1), threads=threads) as (bx, by, bz):
            id_b = bz // Head_V
            id_hv = bz % Head_V
            id_h = id_hv // (Head_V // Head)
            Q_shared = T.alloc_shared([block_K], dtype)
            Q_fragment = T.alloc_fragment([block_K], dtype)
            K_shared = T.alloc_shared([block_K], dtype)
            V_shared = T.alloc_shared([block_V], dtype)
            V_fragment = T.alloc_fragment([block_V], dtype)
            h0_shared = T.alloc_shared([block_K, block_V], accum_dtype)
            o_shared = T.alloc_shared([block_V], dtype)
            
            h0_fragment = T.alloc_fragment([block_K, block_V], accum_dtype)
            K_fragment = T.alloc_fragment([block_K], accum_dtype)
            v_min_reg = T.alloc_fragment([block_V], accum_dtype)

            
            T.copy(h0[id_b, id_hv, bx * block_K, by * block_V], h0_shared)
  
            # T.use_swizzle(8)
            # for t in T.Pipelined(Token, num_stages=num_stages):
            for t in T.serial(Token):
                T.copy(Q[id_b, t, id_h, bx * block_K], Q_shared)
                T.copy(K[id_b, t, id_h, bx * block_K], K_shared)
                T.copy(V[id_b, t, id_hv, by * block_V], V_shared)
 
                # b_q = b_q * scale
                T.copy(Q_shared, Q_fragment)
                T.copy(K_shared, K_fragment)
                T.copy(V_shared, V_fragment)
                T.copy(h0_shared, h0_fragment)
                L2Norm_QK(Q_fragment)
                L2Norm_QK(K_fragment)
                for i in T.Parallel(block_K):
                    Q_fragment[i] = Q_fragment[i] * scale[0]

                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] * T.exp(g[id_b, t, id_hv])
                ### b_v -= tl.sum(b_h * b_k[:, None], 0)
                T.copy(h0_fragment, h0_shared)
                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] * K_fragment[i]
                T.reduce_sum(h0_fragment, v_min_reg, dim=0)
                for i in T.Parallel(block_V):
                    V_fragment[i] = V_fragment[i] - v_min_reg[i]
                ### finished
                ### b_v *= b_beta
                for i in T.Parallel(block_V):
                    V_fragment[i] = V_fragment[i] * Beta[id_b, t, id_hv]
                ### b_h += b_k[:, None] * b_v[None, :]
                T.copy(h0_shared, h0_fragment)
                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] + K_fragment[i] * V_fragment[j]
                ### b_o = tl.sum(b_h * b_q[:, None], 0)
                T.copy(h0_fragment, h0_shared)
                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] * Q_fragment[i]
                ### reuse v_min_reg
                T.reduce_sum(h0_fragment, v_min_reg, dim=0)
                T.copy(v_min_reg, o_shared)
                
                T.copy(o_shared, O[id_b, t, id_hv, by * block_V])
                

            if STORE_FINAL_STATE:
                T.copy(h0_shared, h0[id_b, id_hv, bx * block_K, by * block_V])
        
    return main
#### currently cannot deal with headwise beta(4dim) and store final state is always true
def fused_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, Token, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    device = q.device
    if scale is None:
        scale = K ** -0.5
    o = torch.empty(B, Token, HV, V, device=device, dtype=v.dtype)
    # ht = torch.empty(B, HV, K, V, device=device, dtype=v.dtype)
    if initial_state is None:
        h0 = torch.zeros(B, HV, K, V, device=device, dtype=g.dtype)
    else:
        h0 = initial_state
    kernel = fused_recurrent(B, Token, H, HV, K, V, 
                             use_qk_l2norm_in_kernel, STORE_FINAL_STATE=True)
    # print(kernel.get_kernel_source())
    scale = torch.tensor([scale]).cuda().to(v.dtype)
    torch.cuda.synchronize()
    # print("before kernel")
    kernel(q, k, v, g, beta, o, h0, scale)
    torch.cuda.synchronize()
    # print("after kernel")
    return o, h0
          # [B,T,HV]
class FusedRecurrentFunctionTL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False
    ):
        o, final_state = fused_recurrent_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens
        )
        # ctx.save_for_backward(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, use_qk_l2norm_in_kernel)
        return o, final_state
    
    @staticmethod
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps."
        )

def fused_recurrent_gated_delta_rule_tl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    #### TODO hardcode culength to None
    o, final_state = FusedRecurrentFunctionTL.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        None,
        use_qk_l2norm_in_kernel
    )
    return o, final_state


if __name__ == '__main__':
    # 注释掉原始测试代码
    B, Token, H, HV, K, V = 1, 10, 12, 12, 96, 256
    scale = K ** -0.5
    datatype = torch.bfloat16
    gdtype = torch.float32
    # B = 30
    q = torch.randn(B, Token, H, K, device='cuda', dtype=datatype)
    k = torch.randn(B, Token, H, K, device='cuda', dtype=datatype)
    v = torch.randn(B, Token, HV, V, device='cuda', dtype=datatype)
    g = torch.randn(B, Token, HV, device='cuda', dtype=gdtype).sigmoid()
    beta = torch.randn(B, Token, HV, device='cuda', dtype=datatype).sigmoid()
    h0 = torch.zeros(B, HV, K, V, device='cuda', dtype=gdtype)
    
    o, final_state = fused_recurrent_gated_delta_rule_tl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True
    )
    o_fla, final_state_fla = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True
    )
    
    # 计算输出 o 的误差
    eps = 1e-8
    o_diff = torch.abs(o.float() - o_fla.float())
    o_relative_error = o_diff / (torch.abs(o_fla.float()) + eps)
    o_mean_relative_error = o_relative_error.mean().item()
    o_max_relative_error = o_relative_error.max().item()
    
    # 计算最终状态 final_state 的误差
    final_state_diff = torch.abs(final_state.float() - final_state_fla.float())
    final_state_relative_error = final_state_diff / (torch.abs(final_state_fla.float()) + eps)
    final_state_mean_relative_error = final_state_relative_error.mean().item()
    final_state_max_relative_error = final_state_relative_error.max().item()
    
    # 打印结果
    print("\n" + "="*60)
    print("FLA vs TileLang Kernel 误差对比")
    print("="*60)
    print(f"\n输出 (o) 的误差:")
    print(f"  平均相对误差: {o_mean_relative_error:.6e}")
    print(f"  最大相对误差: {o_max_relative_error:.6e}")
    print(f"  平均绝对误差: {o_diff.mean().item():.6e}")
    print(f"  最大绝对误差: {o_diff.max().item():.6e}")
    
    print(f"\n最终状态 (final_state) 的误差:")
    print(f"  平均相对误差: {final_state_mean_relative_error:.6e}")
    print(f"  最大相对误差: {final_state_max_relative_error:.6e}")
    print(f"  平均绝对误差: {final_state_diff.mean().item():.6e}")
    print(f"  最大绝对误差: {final_state_diff.max().item():.6e}")
    
    print("\n" + "="*60)
    print(f"输出形状: {o.shape}")
    print(f"最终状态形状: {final_state.shape}")
    print("="*60 + "\n")
    
    # for _ in range(10):
    #     o, final_state = fused_recurrent_gated_delta_rule_tl(
    #         q=q,
    #         k=k,
    #         v=v,
    #         g=g,
    #         beta=beta,
    #         scale=scale,
    #         initial_state=h0,
    #         output_final_state=True,
    #         use_qk_l2norm_in_kernel=True
    #     )
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(100):
    #     o, final_state = fused_recurrent_gated_delta_rule_tl(
    #         q=q,
    #         k=k,
    #         v=v,
    #         g=g,
    #         beta=beta,
    #         scale=scale,
    #         initial_state=h0,
    #         output_final_state=True,
    #         use_qk_l2norm_in_kernel=True
    #     )
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"Time: {end - start}")
    