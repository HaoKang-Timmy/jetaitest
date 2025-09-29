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
@autotune(configs=get_configs(), warmup=10, rep=10)
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
    accum_dtype = "float32",
    block_K=128, 
    block_V=128, 
    num_stages=3, 

    threads=256
):
    @T.macro
    def L2Norm_QK(
        QK: T.FragmentBuffer([block_K],dtype),
    ):
        squared_reg = T.alloc_fragment([block_K], dtype)
        sum_reg = T.alloc_fragment([1], dtype)
        
        # 计算元素的平方用于求norm
        for i in T.Parallel(block_K):
            squared_reg[i] = QK[i] * QK[i]
        T.reduce_sum(squared_reg, sum_reg, dim=0)
        sum_reg[0] = T.sqrt(sum_reg[0]) + 1e-6
        
        # 用原始元素除以norm
        for i in T.Parallel(block_K):
            QK[i] = QK[i] / sum_reg[0]


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
            for t in T.serial(Token):
                T.copy(Q[id_b, t, id_h, bx * block_K], Q_shared)
                T.copy(K[id_b, t, id_h, bx * block_K], K_shared)
                T.copy(V[id_b, t, id_hv, by * block_V], V_shared)
 
                # b_q = b_q * scale
                T.copy(Q_shared, Q_fragment)
                T.copy(K_shared, K_fragment)
                T.copy(V_shared, V_fragment)
                T.copy(h0_shared, h0_fragment)
                # if USE_QK_L2NORM_IN_KERNEL:
                #     L2Norm_QK(Q_fragment)
                #     L2Norm_QK(K_fragment)
                # b_q = b_q * scale
                for i in T.Parallel(block_K):
                    Q_fragment[i] = Q_fragment[i] * scale[0]
                #b_h *= exp(b_g)
                
                for i, j in T.Parallel(block_K, block_V):
                    h0_fragment[i, j] = h0_fragment[i, j] * T.exp(g[id_b, t, id_hv])
                T.copy(h0_fragment, h0_shared)
                ### b_v -= tl.sum(b_h * b_k[:, None], 0)

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
                # T.copy(h0_shared, h0_fragment)
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
    device, dtype = q.device, q.dtype
    if scale is None:
        scale = K ** -0.5
    o = torch.empty(B, Token, HV, V, device=device, dtype=v.dtype)
    # ht = torch.empty(B, HV, K, V, device=device, dtype=v.dtype)
    if initial_state is None:
        h0 = torch.zeros(B, HV, K, V, device=device, dtype=torch.float)
    else:
        h0 = initial_state
    kernel = fused_recurrent(B, Token, H, HV, K, V, 
                             use_qk_l2norm_in_kernel, STORE_FINAL_STATE=True)
    scale = torch.tensor([scale]).to(device=q.device, dtype=dtype)
    # print("q dtype:", q.dtype)
    # print("k dtype:", k.dtype)
    # print("v dtype:", v.dtype)
    # print("g dtype:", g.dtype)
    # print("beta dtype:", beta.dtype)
    # print("o dtype:", o.dtype)
    # print("h0 dtype:", h0.dtype)
    # print("scale dtype:", scale.dtype)
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
    B, Token, H, HV, K, V = 1, 10, 1, 1, 128, 128
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
    # benchmark
    # torch.cuda.synchronize()
    # # warm up
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
    #         use_qk_l2norm_in_kernel=False
    #     )
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(20):
    #     o, final_state = fused_recurrent_gated_delta_rule_tl(
    #         q=q,
    #         k=k,
    #         v=v,
    #         g=g,
    #         beta=beta,
    #         scale=scale,
    #         initial_state=h0,
    #         output_final_state=True,
    #         use_qk_l2norm_in_kernel=False
    #     )
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"Batch size: {B}, time: {(end - start) / 100 * 1000} ms")

    # 新的性能测试代码
    # import pandas as pd
    
    # # 固定参数
    # B = 40
    # Token = 1
    # dtype = torch.float16
    
    # # 参数范围
    # head_values = list(range(4, 33, 4))  # [4, 8, 12, 16, 20, 24, 28, 32]
    # k_values = list(range(32, 256, 32))  # [32, 64, 96, ..., 512]
    # v_values = list(range(32, 256, 32))  # [32, 64, 96, ..., 512]
    
    # results = []
    # total_configs = len(head_values) * len(k_values) * len(v_values)
    # current_config = 0
    
    # print(f"开始性能测试，总共 {total_configs} 个配置...")
    # print("参数设置: B={}, Token={}".format(B, Token))
    # print("H/HV范围: {} 到 {}，步长 4".format(min(head_values), max(head_values)))
    # print("K/V范围: {} 到 {}，步长 32".format(min(k_values), max(k_values)))
    # print("-" * 80)
    
    # for H in head_values:
    #     HV = H  # H和HV相等
    #     for K in k_values:
    #         for V in v_values:
    #             current_config += 1
    #             print(f"测试配置 {current_config}/{total_configs}: H={H}, HV={HV}, K={K}, V={V}")
                
    #             try:
    #                 scale = K ** -0.5
                    
    #                 # 创建测试数据
    #                 q = torch.randn(B, Token, H, K, device='cuda', dtype=dtype)
    #                 k = torch.randn(B, Token, H, K, device='cuda', dtype=dtype)
    #                 v = torch.randn(B, Token, HV, V, device='cuda', dtype=dtype)
    #                 g = torch.randn(B, Token, HV, device='cuda', dtype=dtype).sigmoid()
    #                 beta = torch.randn(B, Token, HV, device='cuda', dtype=dtype).sigmoid()
    #                 h0 = torch.zeros(B, HV, K, V, device='cuda', dtype=dtype)
                    
    #                 # 预热
    #                 torch.cuda.synchronize()
    #                 for _ in range(5):
    #                     try:
    #                         o, final_state = fused_recurrent_gated_delta_rule_tl(
    #                             q=q, k=k, v=v, g=g, beta=beta, scale=scale,
    #                             initial_state=h0, output_final_state=True,
    #                             use_qk_l2norm_in_kernel=False
    #                         )
    #                     except Exception as e:
    #                         print(f"预热失败: {e}")
    #                         break
                    
    #                 # 正式测试
    #                 torch.cuda.synchronize()
    #                 start = time.time()
                    
    #                 num_runs = 10
    #                 for _ in range(num_runs):
    #                     o, final_state = fused_recurrent_gated_delta_rule_tl(
    #                         q=q, k=k, v=v, g=g, beta=beta, scale=scale,
    #                         initial_state=h0, output_final_state=True,
    #                         use_qk_l2norm_in_kernel=False
    #                     )
                    
    #                 torch.cuda.synchronize()
    #                 end = time.time()
                    
    #                 avg_time_ms = (end - start) / num_runs * 1000
                    
    #                 results.append({
    #                     'H': H,
    #                     'HV': HV,
    #                     'K': K,
    #                     'V': V,
    #                     'Time_ms': avg_time_ms
    #                 })
                    
    #                 print(f"  完成，平均时间: {avg_time_ms:.3f} ms")
                    
    #                 # 清理内存
    #                 del q, k, v, g, beta, h0, o, final_state
    #                 torch.cuda.empty_cache()
                    
    #             except Exception as e:
    #                 print(f"  配置失败: {e}")
    #                 results.append({
    #                     'H': H,
    #                     'HV': HV,
    #                     'K': K,
    #                     'V': V,
    #                     'Time_ms': -1  # 用-1表示失败
    #                 })
    
    # # 创建DataFrame并保存结果
    # df = pd.DataFrame(results)
    
    # # 保存为CSV文件
    # output_file = "fused_recurrent_performance_results.csv"
    # df.to_csv(output_file, index=False)
    # print(f"\n结果已保存到 {output_file}")
    
    # # 显示汇总统计
    # print("\n=== 性能测试结果汇总 ===")
    # print(f"总配置数: {len(df)}")
    # print(f"成功配置数: {len(df[df['Time_ms'] > 0])}")
    # print(f"失败配置数: {len(df[df['Time_ms'] == -1])}")
    
    # if len(df[df['Time_ms'] > 0]) > 0:
    #     success_df = df[df['Time_ms'] > 0]
    #     print(f"最快时间: {success_df['Time_ms'].min():.3f} ms")
    #     print(f"最慢时间: {success_df['Time_ms'].max():.3f} ms")
    #     print(f"平均时间: {success_df['Time_ms'].mean():.3f} ms")
        
    #     # 显示前10个最快的配置
    #     print("\n前10个最快的配置:")
    #     top_10 = success_df.nsmallest(10, 'Time_ms')
    #     for idx, row in top_10.iterrows():
    #         print(f"  H={row['H']}, K={row['K']}, V={row['V']}: {row['Time_ms']:.3f} ms")
    
    # print("\n完整结果表格:")
    # print(df.to_string(index=False))