import torch
import tilelang
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
    threads = [128, 256]
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
@tilelang.jit
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
    accum_dtype = "float",
    block_K=128, 
    block_V=128, 
    num_stages=2, 

    threads=128
):
    @T.macro
    def L2Norm_QK(
        QK: T.SharedBuffer([block_K],accum_dtype),  # 改为float32测试
    ):
        shared_reg = T.alloc_fragment([block_K], accum_dtype)  # 改为float32测试
        squared_reg = T.alloc_fragment([block_K], accum_dtype)  # 改为float32测试
        sum_reg = T.alloc_fragment([1], accum_dtype)  # 改为float32测试
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
            Q_shared = T.alloc_shared([block_K], dtype)  # 改为float32测试
            K_shared = T.alloc_shared([block_K], dtype)  # 改为float32测试
            V_shared = T.alloc_shared([block_V], dtype)  # 改为float32测试
            h0_shared = T.alloc_shared([block_K, block_V], accum_dtype)
            o_shared = T.alloc_shared([block_V], dtype)
            
            scale_reg = T.alloc_fragment([1], accum_dtype)  # 改为float32测试
            g_reg = T.alloc_local([1], accum_dtype)  # 改为float32测试
            beta_reg = T.alloc_local([1], accum_dtype)  # 改为float32测试
            h0_reg = T.alloc_fragment([block_K, block_V], accum_dtype)
            k_reg = T.alloc_fragment([block_K], accum_dtype)  # 改为float32测试
            v_min_reg = T.alloc_fragment([block_V], accum_dtype)  # 改为float32测试
            
            scale_reg[0] = scale[0]
            
            T.copy(h0[id_b, id_hv, bx * block_K, by * block_V], h0_shared)
            # T.annotate_layout(
            #     {
            #         Q_shared: tl.layout.make_swizzled_layout(Q_shared),
            #         K_shared: tl.layout.make_swizzled_layout(K_shared),
            #         V_shared: tl.layout.make_swizzled_layout(V_shared),
            #         h0_shared: tl.layout.make_swizzled_layout(h0_shared),
            #         o_shared: tl.layout.make_swizzled_layout(o_shared),
            #     }
            # )
            # T.use_swizzle(8)
            # for t in T.Pipelined(Token, num_stages=num_stages):
            for t in T.serial(Token):
                T.copy(Q[id_b, t, id_h, bx * block_K], Q_shared)
                T.copy(K[id_b, t, id_h, bx * block_K], K_shared)
                T.copy(V[id_b, t, id_hv, by * block_V], V_shared)
                # T.copy(O[id_b, t, id_hv, by * block_V], o_shared)
                # elementwise copy
                beta_reg[0] = Beta[id_b, t, id_hv]
                g_reg[0] = g[id_b, t, id_hv]
                if USE_QK_L2NORM_IN_KERNEL:
                    L2Norm_QK(Q_shared)
                    L2Norm_QK(K_shared)
                # b_q = b_q * scale
                for i in T.Parallel(block_K):
                    Q_shared[i] = Q_shared[i] * scale_reg[0]
                #b_h *= exp(b_g)
                
                T.copy(h0_shared, h0_reg)
                for i, j in T.Parallel(block_K, block_V):
                    h0_reg[i, j] = h0_reg[i, j] * T.exp(g_reg[0])
                T.copy(h0_reg, h0_shared)
                ### b_v -= tl.sum(b_h * b_k[:, None], 0)
                
                T.copy(K_shared, k_reg)
                for i, j in T.Parallel(block_K, block_V):
                    h0_reg[i, j] = h0_reg[i, j] * k_reg[i]
                T.reduce_sum(h0_reg, v_min_reg, dim=0)
                for i in T.Parallel(block_V):
                    V_shared[i] = V_shared[i] - v_min_reg[i]
                ### finished
                ### b_v *= b_beta
                for i in T.Parallel(block_V):
                    V_shared[i] = V_shared[i] * beta_reg[0]
                ### b_h += b_k[:, None] * b_v[None, :]
                for i, j in T.Parallel(block_K, block_V):
                    h0_shared[i, j] = h0_shared[i, j] + K_shared[i] * V_shared[j]
                ### b_o = tl.sum(b_h * b_q[:, None], 0)
                T.copy(h0_shared, h0_reg)
                for i, j in T.Parallel(block_K, block_V):
                    h0_reg[i, j] = h0_reg[i, j] * Q_shared[i]
                ### reuse v_min_reg
                T.reduce_sum(h0_reg, v_min_reg, dim=0)
                T.copy(v_min_reg, o_shared)
                
                T.copy(o_shared, O[id_b, t, id_hv, by * block_V])
                

            if STORE_FINAL_STATE:
                T.copy(h0_shared, h0[id_b, id_hv, bx * block_K, by * block_V])
        
    return main


def fused_recurrent_python_reference(
    Q: torch.Tensor,  # [B, T, H, K]
    K: torch.Tensor,  # [B, T, H, K] 
    V: torch.Tensor,  # [B, T, HV, V]
    g: torch.Tensor,  # [B, T, HV]
    Beta: torch.Tensor,  # [B, T, HV]
    scale: torch.Tensor,  # [1]
    h0: torch.Tensor,  # [B, HV, K, V]
    USE_QK_L2NORM_IN_KERNEL: bool = True,
    STORE_FINAL_STATE: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TileLang kernel的完全等价Python实现
    用于调试和验证TileLang kernel的正确性
    """
    B, T, H, K_dim = Q.shape  # 修复：用K_dim避免与参数K冲突
    _, _, HV, V_dim = V.shape  # 修复：用V_dim避免与参数V冲突
    device = Q.device
    
    # 输出tensor
    O = torch.zeros(B, T, HV, V_dim, device=device, dtype=Q.dtype)
    
    # 为每个batch和head处理
    for b in range(B):
        for hv in range(HV):
            h = hv // (HV // H)  # head index for Q,K
            
            # 初始化hidden state (对应TileLang中的h0_shared)
            h_state = h0[b, hv].clone().float()  # [K_dim, V_dim]
            
            # 对每个时间步处理
            for t in range(T):
                # 加载当前时间步的数据
                q_t = Q[b, t, h].float()  # [K_dim] 
                k_t = K[b, t, h].float()  # [K_dim]  # 现在K是tensor参数，不是维度
                v_t = V[b, t, hv].float()  # [V_dim]
                g_t = g[b, t, hv].float()  # scalar
                beta_t = Beta[b, t, hv].float()  # scalar
                
                # L2归一化 (如果启用)
                if USE_QK_L2NORM_IN_KERNEL:
                    q_norm = torch.sqrt(torch.sum(q_t * q_t)) + 1e-6
                    q_t = q_t / q_norm
                    
                    k_norm = torch.sqrt(torch.sum(k_t * k_t)) + 1e-6  
                    k_t = k_t / k_norm
                
                # 缩放q
                q_t = q_t * scale.float()
                
                # 状态衰减: h_state *= exp(g_t)
                h_state = h_state * torch.exp(g_t)
                
                # 计算v的修正: v_t -= sum(h_state * k_t[:, None], dim=0)
                v_correction = torch.sum(h_state * k_t[:, None], dim=0)  # [V]
                v_t = v_t - v_correction
                
                # 应用beta: v_t *= beta_t
                v_t = v_t * beta_t
                
                # 更新状态: h_state += k_t[:, None] * v_t[None, :]
                h_state = h_state + k_t[:, None] * v_t[None, :]
                
                # 计算输出: o_t = sum(h_state * q_t[:, None], dim=0)
                o_t = torch.sum(h_state * q_t[:, None], dim=0)  # [V]
                
                # 存储输出
                O[b, t, hv] = o_t.to(Q.dtype)
            
            # 存储最终状态
            if STORE_FINAL_STATE:
                h0[b, hv] = h_state.to(h0.dtype)
    
    return O, h0
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
        h0 = torch.zeros(B, HV, K, V, device=device, dtype=v.dtype)
    else:
        h0 = initial_state
    kernel = fused_recurrent(B, Token, H, HV, K, V, 
                             use_qk_l2norm_in_kernel, STORE_FINAL_STATE=True)
    scale = torch.tensor([scale]).cuda().to(v.dtype)
    kernel(q, k, v, g, beta, o, h0, scale)
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
    print("=== 测试TileLang vs Triton实现差异 ===")
    print("注意：已将TileLang中Q、K、V等变量类型改为float32，以排除混合精度影响")
    
    # 设置随机种子确保结果可重现
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 测试参数
    B, Token, H, HV, K, V = 2, 10, 1, 1, 128, 128
    scale = K ** -0.5
    datatype = torch.bfloat16
    gdtype = torch.float32
    
    # 生成固定的测试数据
    q = torch.randn(B, Token, H, K, device='cuda', dtype=datatype)
    k = torch.randn(B, Token, H, K, device='cuda', dtype=datatype)
    v = torch.randn(B, Token, HV, V, device='cuda', dtype=datatype)
    g = torch.randn(B, Token, HV, device='cuda', dtype=gdtype).sigmoid()
    beta = torch.randn(B, Token, HV, device='cuda', dtype=datatype).sigmoid()
    h0 = torch.zeros(B, HV, K, V, device='cuda', dtype=gdtype)
    h0_clone1 = h0.clone()
    h0_clone2 = h0.clone()
    
    print("\n1. 运行TileLang kernel实现...")
    o_tilelang, final_state_tilelang = fused_recurrent_gated_delta_rule_tl(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True
    )
    
    print("2. 运行Triton kernel实现...")
    o_triton, final_state_triton = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0_clone1,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True
    )
    
    print("3. 运行Python参考实现...")
    scale_tensor = torch.tensor([scale]).cuda().to(datatype)
    o_python, final_state_python = fused_recurrent_python_reference(
        Q=q,
        K=k,
        V=v,
        g=g,
        Beta=beta,
        scale=scale_tensor,
        h0=h0_clone2,
        USE_QK_L2NORM_IN_KERNEL=True,
        STORE_FINAL_STATE=True
    )

    def calculate_relative_errors(actual, expected, eps=1e-10):
        """
        计算平均相对误差和最大相对误差
        对于0值的处理：当expected为0时，如果actual也接近0则误差为0，否则使用绝对误差
        """
        actual_flat = actual.flatten()
        expected_flat = expected.flatten()
        
        # 计算绝对误差
        abs_diff = torch.abs(actual_flat - expected_flat)
        
        # 计算相对误差，对0值进行特殊处理
        expected_abs = torch.abs(expected_flat)
        
        # 当expected不为0时，使用相对误差
        non_zero_mask = expected_abs > eps
        relative_errors = torch.zeros_like(abs_diff)
        relative_errors[non_zero_mask] = abs_diff[non_zero_mask] / expected_abs[non_zero_mask]
        
        # 当expected为0时，如果actual也接近0则误差为0，否则使用绝对误差
        zero_mask = ~non_zero_mask
        actual_abs = torch.abs(actual_flat)
        relative_errors[zero_mask] = torch.where(
            actual_abs[zero_mask] <= eps, 
            torch.zeros_like(actual_abs[zero_mask]), 
            actual_abs[zero_mask]
        )
        
        # 计算平均相对误差和最大相对误差
        mean_rel_error = torch.mean(relative_errors).item()
        max_rel_error = torch.max(relative_errors).item()
        
        return mean_rel_error, max_rel_error

    print("\n=== 三种实现结果对比 ===")
    
    # 1. TileLang vs Python参考实现
    print("\n📊 TileLang vs Python参考实现:")
    o_mean_rel_err, o_max_rel_err = calculate_relative_errors(o_tilelang, o_python)
    print(f"  输出 'o' 的平均相对误差: {o_mean_rel_err:.6e}")
    print(f"  输出 'o' 的最大相对误差: {o_max_rel_err:.6e}")
    
    fs_mean_rel_err, fs_max_rel_err = calculate_relative_errors(
        final_state_tilelang, final_state_python.to(final_state_tilelang.dtype)
    )
    print(f"  final_state 的平均相对误差: {fs_mean_rel_err:.6e}")
    print(f"  final_state 的最大相对误差: {fs_max_rel_err:.6e}")
    
    # 2. Triton vs Python参考实现
    print("\n📊 Triton vs Python参考实现:")
    o_mean_rel_err, o_max_rel_err = calculate_relative_errors(o_triton, o_python)
    print(f"  输出 'o' 的平均相对误差: {o_mean_rel_err:.6e}")
    print(f"  输出 'o' 的最大相对误差: {o_max_rel_err:.6e}")
    
    fs_mean_rel_err, fs_max_rel_err = calculate_relative_errors(
        final_state_triton.to(final_state_python.dtype), final_state_python
    )
    print(f"  final_state 的平均相对误差: {fs_mean_rel_err:.6e}")
    print(f"  final_state 的最大相对误差: {fs_max_rel_err:.6e}")
    
    # 3. TileLang vs Triton实现
    print("\n📊 TileLang vs Triton实现:")
    o_mean_rel_err, o_max_rel_err = calculate_relative_errors(o_tilelang, o_triton)
    print(f"  输出 'o' 的平均相对误差: {o_mean_rel_err:.6e}")
    print(f"  输出 'o' 的最大相对误差: {o_max_rel_err:.6e}")
    
    fs_mean_rel_err, fs_max_rel_err = calculate_relative_errors(
        final_state_tilelang, final_state_triton.to(final_state_tilelang.dtype)
    )
    print(f"  final_state 的平均相对误差: {fs_mean_rel_err:.6e}")
    print(f"  final_state 的最大相对误差: {fs_max_rel_err:.6e}")
    
    # # 样本值对比
    # print("\n📋 样本值对比 (前5个元素):")
    # print("TileLang 输出 o:", o_tilelang.flatten()[:5])
    # print("Triton   输出 o:", o_triton.flatten()[:5])
    # print("Python   输出 o:", o_python.flatten()[:5])
    # print()
    # print("TileLang final_state:", final_state_tilelang.flatten()[:5])
    # print("Triton   final_state:", final_state_triton.flatten()[:5])
    # print("Python   final_state:", final_state_python.flatten()[:5])
