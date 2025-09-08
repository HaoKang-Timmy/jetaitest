import torch
import tilelang
from tilelang.autotuner import autotune
import tilelang.language as T
import itertools
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_configs():
    block_D = [128, 256, 512]
    threads = [128, 256]
    _configs = list(itertools.product(block_D, threads))
    configs = [{
        'block_D': c[0],
        'threads': c[1]
    } for c in _configs]
    return configs

@autotune(configs=get_configs(), warmup=10, rep=10)
@tilelang.jit(
    # pass_configs={
    #     tilelang.PassConfigKey.TL_DISABLE_FAST_MATH: False,
    # }
)
def _dconv_fwd_kernel(
    Batch,
    Token,
    Indim,
    Kernel_size,
    block_D,
    threads,
    dtype,  
    reduce_type = "float32"
):
    @T.prim_func
    def main_fp16_prefill(
        Input: T.Tensor([Batch * Token, Indim], dtype),
        Kernel_input: T.Tensor([Batch * Token, Indim, Kernel_size], dtype),
        Output: T.Tensor([Batch * Token, Indim], dtype),
        # Cache: T.Tensor([Batch * Token, T_cache], dtype),
    ):
        with T.Kernel(Batch * Token, T.ceildiv(Indim, block_D), threads=threads) as (bx, by):
            Kernel_shared = T.alloc_shared([block_D, Kernel_size], dtype)
            Input_shared = T.alloc_shared([block_D, Kernel_size], dtype)
            Output_shared = T.alloc_shared([block_D], dtype)
            Input_reg = T.alloc_fragment([block_D, Kernel_size], reduce_type)
            Output_reg = T.alloc_fragment([block_D, Kernel_size], reduce_type)
            Output_reduced = T.alloc_fragment([block_D], reduce_type)
            Kernel_reg = T.alloc_fragment([block_D, Kernel_size], reduce_type)
            batch_id = bx // Token
            token_id = bx % Token
            # T.copy(Kernel_input[bx, by * block_D,0], Kernel_shared)
            for i, j in T.Parallel(block_D, Kernel_size):
                Kernel_shared[i, j] = Kernel_input[bx, i + by * block_D, j]
            ## copy input
            for i, j in T.Parallel(block_D, Kernel_size):
                Input_shared[i, j] = T.if_then_else(token_id + 1 + j - Kernel_size >= 0, 
                                                  Input[bx - Kernel_size + 1 + j, i + by * block_D], 0.0)
            
            T.copy(Input_shared, Input_reg)
            T.copy(Kernel_shared, Kernel_reg)
            for i, j in T.Parallel(block_D, Kernel_size):
                Output_reg[i, j] = Input_reg[i, j] * Kernel_reg[i, j]  
            T.reduce_sum(Output_reg, Output_reduced, dim=1)
            T.copy(Output_reduced, Output_shared)
            T.copy(Output_shared, Output[bx, by * block_D])
    return main_fp16_prefill

# def dynamic_conv_pytorch_cache(x, kernels, cache=None):
#     B, T, D = x.shape
#     W = kernels.shape[-1]
#     T_CACHE = cache.shape[1] if cache is not None else 0
    
#     output = torch.zeros_like(x)
    
#     for b in range(B):
#         for t in range(T):
#             for d in range(D):
#                 accumulator = 0.0
                
#                 for w in range(W):
#                     input_time_rel_x = t + w - W + 1
#                     eff_time_idx = input_time_rel_x + T_CACHE
                    
#                     if eff_time_idx >= 0 and eff_time_idx < (T_CACHE + T):
#                         if eff_time_idx < T_CACHE:
#                             cache_idx = eff_time_idx
#                             input_val = cache[b, cache_idx, d].item()
#                         else:
#                             x_idx = eff_time_idx - T_CACHE
#                             input_val = x[b, x_idx, d].item()
#                     else:
#                         input_val = 0.0
                    
#                     kernel_val = kernels[b, t, d, w].item()
#                     accumulator += input_val * kernel_val
                
#                 output[b, t, d] = accumulator
    
#     return output

if __name__ == "__main__":
    batch = 4
    token = 10
    indim = 512
    kernel_size = 4
    block_d = 32
    num_stages = 1
    threads = 128
    dtype = "float16"
    _dconv_fwd_kernel(batch, token, indim, kernel_size, dtype = dtype)
    input1 = torch.randn(batch * token, indim).to(torch.float16).cuda()
    kernel_input = torch.randn(batch * token, indim, kernel_size).to(torch.float16).cuda()
    output_triton = torch.randn(batch * token, indim).to(torch.float16).cuda()
    input1 = input1 * 5
    kernel_input = kernel_input * 5
    kernel = _dconv_fwd_kernel(batch, token, indim, kernel_size, block_d, threads, dtype)
    kernel(input1, kernel_input, output_triton)
    # print("output_triton",output_triton)
    # Reshape inputs for PyTorch version
    # x = input1.view(batch, token, indim)
    # kernels = kernel_input.view(batch, token, indim, kernel_size)
    
    # # Run PyTorch version with cache=None
    # output_pytorch = dynamic_conv_pytorch_cache(x, kernels, cache=None)
    # # print("output_pytorch",output_pytorch)
    # # Reshape back if necessary
    # output_pytorch = output_pytorch.view(batch * token, indim)
    
    # # Compute relative error
    # abs_diff = torch.abs(output_pytorch - output_triton)
    # rel_diff = abs_diff / (torch.abs(output_pytorch) + 1e-8)
    # max_rel_diff = torch.max(rel_diff).item()
    # mean_rel_diff = torch.mean(rel_diff).item()
    
    # # Compute absolute error
    # max_abs_diff = torch.max(abs_diff).item()
    # mean_abs_diff = torch.mean(abs_diff).item()
    
    # print(f"最大相对误差: {max_rel_diff}")
    # print(f"平均相对误差: {mean_rel_diff}")
    # print(f"最大绝对误差: {max_abs_diff}")
    # print(f"平均绝对误差: {mean_abs_diff}")
    
    # # Count elements with relative error above average
    # above_avg_count = torch.sum(rel_diff > mean_rel_diff).item()
    # total_elements = rel_diff.numel()
    # above_avg_percentage = (above_avg_count / total_elements) * 100
    
    # print(f"相对误差超过平均值的元素个数: {above_avg_count} / {total_elements} ({above_avg_percentage:.2f}%)")
    
    # # Check if close
    # is_close = torch.allclose(output_pytorch, output_triton, rtol=1e-4, atol=1e-5)
    # print(f"结果是否匹配 (rtol=1e-4, atol=1e-5): {is_close}")
    
    # # 可视化相对误差热力图
    # print("生成相对误差热力图...")
    
    # # 将相对误差转换为numpy数组并重新整形
    # rel_diff_cpu = rel_diff.cpu().numpy()
    # rel_diff_reshaped = rel_diff_cpu.reshape(batch, token, indim)
    
    # # 创建图形
    # fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    # fig.suptitle('相对误差可视化分析', fontsize=16, fontweight='bold')
    
    # # 1. 整体相对误差热力图 (batch*token vs indim)
    # ax1 = axes[0, 0]
    # im1 = ax1.imshow(rel_diff_cpu.reshape(batch * token, indim), 
    #                  cmap='viridis', aspect='auto', interpolation='nearest')
    # ax1.set_title('整体相对误差热力图\n(Batch×Token vs Input_Dimension)')
    # ax1.set_xlabel('输入维度 (Input Dimension)')
    # ax1.set_ylabel('批次×时间步 (Batch × Token)')
    # plt.colorbar(im1, ax=ax1, label='相对误差')
    
    # # 2. 平均相对误差 - 按维度
    # ax2 = axes[0, 1]
    # mean_error_per_dim = np.mean(rel_diff_reshaped, axis=(0, 1))
    # ax2.plot(range(indim), mean_error_per_dim, 'b-', linewidth=2)
    # ax2.set_title('各维度平均相对误差')
    # ax2.set_xlabel('输入维度 (Input Dimension)')
    # ax2.set_ylabel('平均相对误差')
    # ax2.grid(True, alpha=0.3)
    
    # # 3. 平均相对误差 - 按时间步
    # ax3 = axes[1, 0]
    # mean_error_per_token = np.mean(rel_diff_reshaped, axis=(0, 2))
    # ax3.plot(range(token), mean_error_per_token, 'r-', linewidth=2, marker='o')
    # ax3.set_title('各时间步平均相对误差')
    # ax3.set_xlabel('时间步 (Token)')
    # ax3.set_ylabel('平均相对误差')
    # ax3.grid(True, alpha=0.3)
    
    # # 4. 相对误差分布直方图
    # ax4 = axes[1, 1]
    # ax4.hist(rel_diff_cpu.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    # ax4.set_title('相对误差分布直方图')
    # ax4.set_xlabel('相对误差值')
    # ax4.set_ylabel('频次')
    # ax4.axvline(mean_rel_diff, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_rel_diff:.2e}')
    # ax4.axvline(max_rel_diff, color='orange', linestyle='--', linewidth=2, label=f'最大值: {max_rel_diff:.2e}')
    # ax4.legend()
    # ax4.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.savefig('relative_error_heatmap.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    # # 额外生成详细的热力图 - 每个batch单独显示
    # if batch <= 4:  # 只在batch数量不太大时显示
    #     fig2, axes2 = plt.subplots(1, batch, figsize=(5*batch, 4))
    #     if batch == 1:
    #         axes2 = [axes2]
        
    #     fig2.suptitle('各批次相对误差热力图 (Token vs Input_Dimension)', fontsize=14)
        
    #     for b in range(batch):
    #         im = axes2[b].imshow(rel_diff_reshaped[b], cmap='viridis', aspect='auto')
    #         axes2[b].set_title(f'批次 {b+1}')
    #         axes2[b].set_xlabel('输入维度')
    #         axes2[b].set_ylabel('时间步')
    #         plt.colorbar(im, ax=axes2[b], label='相对误差')
        
    #     plt.tight_layout()
    #     plt.savefig('relative_error_per_batch.png', dpi=300, bbox_inches='tight')
    #     plt.show()
    
    # print(f"热力图已保存为 'relative_error_heatmap.png' 和 'relative_error_per_batch.png'")
    # print(f"可视化完成！")
            
            
        