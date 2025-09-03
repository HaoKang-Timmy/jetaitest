import torch
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
from typing import Optional, Tuple
import time
import tilelang as tl

def get_configs():
    block_K = [32,64,128,256]
    block_V = [32,64,128,256]
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
    dtype = "float16",
    block_K=128, 
    block_V=128, 
    num_stages=2, 

    threads=128
):
    @T.macro
    def L2Norm_QK(
        QK: T.SharedBuffer([block_K],dtype),
    ):
        shared_reg = T.alloc_fragment([block_K], dtype)
        squared_reg = T.alloc_fragment([block_K], dtype)
        sum_reg = T.alloc_fragment([1], dtype)
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
        g: T.Tensor([Batch, Token, Head_V], dtype),
        Beta: T.Tensor([Batch, Token, Head_V],dtype),
        O: T.Tensor([Batch, Token, Head_V, V_Dim], dtype),
        h0: T.Tensor([Batch, Head_V, K_Dim, V_Dim], dtype),
        # ht: T.Tensor([Batch, Head_V, K_Dim, V_Dim], dtype),
        scale: T.Tensor([1], dtype),
        # O: T.Tensor(Batch, Token, Head_V, V_Dim, dtype),
    ):
        with T.Kernel(T.ceildiv(K_Dim, block_K), T.ceildiv(V_Dim, block_V), T.ceildiv(Batch*Head_V, 1), threads=threads) as (bx, by, bz):
            id_b = bz // Head_V
            id_hv = bz % Head_V
            id_h = id_hv // (Head_V // Head)
            Q_shared = T.alloc_shared([block_K], dtype)
            K_shared = T.alloc_shared([block_K], dtype)
            V_shared = T.alloc_shared([block_V], dtype)
            h0_shared = T.alloc_shared([block_K, block_V], dtype)
            o_shared = T.alloc_shared([block_V], dtype)
            
            scale_reg = T.alloc_fragment([1], dtype)
            g_reg = T.alloc_local([1], dtype)
            beta_reg = T.alloc_local([1], dtype)
            h0_reg = T.alloc_fragment([block_K, block_V], dtype)
            k_reg = T.alloc_fragment([block_K], dtype)
            v_min_reg = T.alloc_fragment([block_V], dtype)
            
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
            for t in T.Pipelined(Token, num_stages=num_stages):
                T.copy(Q[id_b, t, id_h, bx * block_K], Q_shared)
                T.copy(K[id_b, t, id_h, bx * block_K], K_shared)
                T.copy(V[id_b, t, id_hv, by * block_V], V_shared)
                T.copy(O[id_b, t, id_hv, by * block_V], o_shared)
                # elementwise copy
                beta_reg[0] = Beta[id_b, t, id_hv]
                g_reg[0] = g[id_b, t, id_hv]
                # if USE_QK_L2NORM_IN_KERNEL:
                #     L2Norm_QK(Q_shared)
                #     L2Norm_QK(K_shared)
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
        h0 = torch.zeros(B, HV, K, V, device=device, dtype=torch.float16)
    else:
        h0 = initial_state
    kernel = fused_recurrent(B, Token, H, HV, K, V, 
                             use_qk_l2norm_in_kernel, STORE_FINAL_STATE=True)
    scale = torch.tensor([scale]).cuda().half()
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
    # 注释掉原始测试代码
    # B, Token, H, HV, K, V = 100, 1, 12, 12, 96, 256
    # scale = K ** -0.5
    # dtype = torch.float16
    # 
    # # B = 30
    # q = torch.randn(B, Token, H, K, device='cuda', dtype=dtype)
    # k = torch.randn(B, Token, H, K, device='cuda', dtype=dtype)
    # v = torch.randn(B, Token, HV, V, device='cuda', dtype=dtype)
    # g = torch.randn(B, Token, HV, device='cuda', dtype=dtype).sigmoid()
    # beta = torch.randn(B, Token, HV, device='cuda', dtype=dtype).sigmoid()
    # h0 = torch.zeros(B, HV, K, V, device='cuda', dtype=dtype)
    # 
    # o, final_state = fused_recurrent_gated_delta_rule_tl(
    #     q=q,
    #     k=k,
    #     v=v,
    #     g=g,
    #     beta=beta,
    #     scale=scale,
    #     initial_state=h0,
    #     output_final_state=True,
    #     use_qk_l2norm_in_kernel=False
    # )
    # 
    # # benchmark
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
    import pandas as pd
    
    # 固定参数
    B = 40
    Token = 1
    dtype = torch.float16
    
    # 参数范围
    head_values = list(range(4, 33, 4))  # [4, 8, 12, 16, 20, 24, 28, 32]
    k_values = list(range(32, 256, 32))  # [32, 64, 96, ..., 512]
    v_values = list(range(32, 256, 32))  # [32, 64, 96, ..., 512]
    
    results = []
    total_configs = len(head_values) * len(k_values) * len(v_values)
    current_config = 0
    
    print(f"开始性能测试，总共 {total_configs} 个配置...")
    print("参数设置: B={}, Token={}".format(B, Token))
    print("H/HV范围: {} 到 {}，步长 4".format(min(head_values), max(head_values)))
    print("K/V范围: {} 到 {}，步长 32".format(min(k_values), max(k_values)))
    print("-" * 80)
    
    for H in head_values:
        HV = H  # H和HV相等
        for K in k_values:
            for V in v_values:
                current_config += 1
                print(f"测试配置 {current_config}/{total_configs}: H={H}, HV={HV}, K={K}, V={V}")
                
                try:
                    scale = K ** -0.5
                    
                    # 创建测试数据
                    q = torch.randn(B, Token, H, K, device='cuda', dtype=dtype)
                    k = torch.randn(B, Token, H, K, device='cuda', dtype=dtype)
                    v = torch.randn(B, Token, HV, V, device='cuda', dtype=dtype)
                    g = torch.randn(B, Token, HV, device='cuda', dtype=dtype).sigmoid()
                    beta = torch.randn(B, Token, HV, device='cuda', dtype=dtype).sigmoid()
                    h0 = torch.zeros(B, HV, K, V, device='cuda', dtype=dtype)
                    
                    # 预热
                    torch.cuda.synchronize()
                    for _ in range(5):
                        try:
                            o, final_state = fused_recurrent_gated_delta_rule_tl(
                                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                                initial_state=h0, output_final_state=True,
                                use_qk_l2norm_in_kernel=False
                            )
                        except Exception as e:
                            print(f"预热失败: {e}")
                            break
                    
                    # 正式测试
                    torch.cuda.synchronize()
                    start = time.time()
                    
                    num_runs = 10
                    for _ in range(num_runs):
                        o, final_state = fused_recurrent_gated_delta_rule_tl(
                            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                            initial_state=h0, output_final_state=True,
                            use_qk_l2norm_in_kernel=False
                        )
                    
                    torch.cuda.synchronize()
                    end = time.time()
                    
                    avg_time_ms = (end - start) / num_runs * 1000
                    
                    results.append({
                        'H': H,
                        'HV': HV,
                        'K': K,
                        'V': V,
                        'Time_ms': avg_time_ms
                    })
                    
                    print(f"  完成，平均时间: {avg_time_ms:.3f} ms")
                    
                    # 清理内存
                    del q, k, v, g, beta, h0, o, final_state
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"  配置失败: {e}")
                    results.append({
                        'H': H,
                        'HV': HV,
                        'K': K,
                        'V': V,
                        'Time_ms': -1  # 用-1表示失败
                    })
    
    # 创建DataFrame并保存结果
    df = pd.DataFrame(results)
    
    # 保存为CSV文件
    output_file = "fused_recurrent_performance_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n结果已保存到 {output_file}")
    
    # 显示汇总统计
    print("\n=== 性能测试结果汇总 ===")
    print(f"总配置数: {len(df)}")
    print(f"成功配置数: {len(df[df['Time_ms'] > 0])}")
    print(f"失败配置数: {len(df[df['Time_ms'] == -1])}")
    
    if len(df[df['Time_ms'] > 0]) > 0:
        success_df = df[df['Time_ms'] > 0]
        print(f"最快时间: {success_df['Time_ms'].min():.3f} ms")
        print(f"最慢时间: {success_df['Time_ms'].max():.3f} ms")
        print(f"平均时间: {success_df['Time_ms'].mean():.3f} ms")
        
        # 显示前10个最快的配置
        print("\n前10个最快的配置:")
        top_10 = success_df.nsmallest(10, 'Time_ms')
        for idx, row in top_10.iterrows():
            print(f"  H={row['H']}, K={row['K']}, V={row['V']}: {row['Time_ms']:.3f} ms")
    
    print("\n完整结果表格:")
    print(df.to_string(index=False))