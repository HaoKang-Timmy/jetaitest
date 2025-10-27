import torch
import numpy as np
import time
from tril_compute import tl_merge_16x16_to_64x64_inverse_kernel, tl_merge_16x16_to_64x64_inverse_pytorch


def compute_relative_errors(output_tvm, output_pytorch):
    """
    计算两个输出之间的相对误差
    
    Args:
        output_tvm: TVM kernel的输出
        output_pytorch: PyTorch实现的输出
    
    Returns:
        mean_rel_error: 平均相对误差
        max_rel_error: 最大相对误差
    """
    # 转换为numpy数组便于计算
    tvm_np = output_tvm.cpu().numpy()
    pytorch_np = output_pytorch.cpu().numpy()
    
    # 计算绝对误差
    abs_diff = np.abs(tvm_np - pytorch_np)
    
    # 计算相对误差（避免除以零）
    # 相对误差 = |a - b| / (|b| + epsilon)
    epsilon = 1e-8
    rel_error = abs_diff / (np.abs(pytorch_np) + epsilon)
    
    # 计算统计量
    mean_rel_error = np.mean(rel_error)
    max_rel_error = np.max(rel_error)
    mean_abs_error = np.mean(abs_diff)
    max_abs_error = np.max(abs_diff)
    
    return {
        'mean_relative_error': mean_rel_error,
        'max_relative_error': max_rel_error,
        'mean_absolute_error': mean_abs_error,
        'max_absolute_error': max_abs_error
    }


def benchmark_function(func, *args, warmup=5, repeats=20):
    """
    测试函数性能
    
    Args:
        func: 要测试的函数
        args: 函数参数
        warmup: 预热次数
        repeats: 重复次数
    
    Returns:
        平均执行时间（毫秒）
    """
    # 预热
    for _ in range(warmup):
        func(*args)
    
    # 同步CUDA
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 计时
    times = []
    for _ in range(repeats):
        start = time.time()
        func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)  # 转换为毫秒
    
    return np.mean(times), np.std(times)


def generate_test_data(B, Token, H, BT, device='cuda', dtype=torch.float32):
    """
    生成测试数据
    
    Args:
        B: batch size
        Token: token数量（必须是64的倍数）
        H: head数量
        BT: A矩阵的最后一维大小
        device: 设备
        dtype: 数据类型
    
    Returns:
        A, Ad: 输入张量
    """
    # 生成随机数据
    A = torch.randn(B, Token, H, BT, device=device, dtype=dtype) * 0.1
    Ad = torch.randn(B, Token, H, 16, device=device, dtype=dtype) * 0.1
    
    # 为了数值稳定性，让Ad的对角块更接近单位矩阵
    # 这样逆运算会更稳定
    for i_t in range(Token // 64):
        for block_idx in range(4):
            start_idx = i_t * 64 + block_idx * 16
            end_idx = start_idx + 16
            # 添加单位矩阵成分使其更稳定
            Ad[:, start_idx:end_idx, :, :] += torch.eye(16, device=device, dtype=dtype).view(1, 16, 1, 16)
    
    return A, Ad


def compare_implementations(B, Token, H, BT, device='cuda', dtype=torch.float32, run_benchmark=True):
    """
    比较TVM和PyTorch实现
    
    Args:
        B: batch size
        Token: token数量（必须是64的倍数）
        H: head数量
        BT: A矩阵的最后一维大小
        device: 设备
        dtype: 数据类型
        run_benchmark: 是否运行性能测试
    """
    print(f"\n{'='*60}")
    print(f"测试配置: B={B}, Token={Token}, H={H}, BT={BT}, dtype={dtype}")
    print(f"{'='*60}")
    
    # 生成测试数据
    print("\n生成测试数据...")
    A, Ad = generate_test_data(B, Token, H, BT, device=device, dtype=dtype)
    
    # 计算NT
    NT = Token // 64
    
    # 运行TVM kernel
    print("\n运行TVM kernel...")
    try:
        tvm_kernel = tl_merge_16x16_to_64x64_inverse_kernel(
            B=B, Token=Token, H=H, BT=BT, NT=NT,
            input_dtype="float32" if dtype == torch.float32 else "float16",
            accum_dtype="float32",
            output_dtype="float32" if dtype == torch.float32 else "float16"
        )
        
        Ai_tvm = torch.zeros(B, Token, H, 64, device=device, dtype=dtype)
        tvm_kernel(A, Ad, Ai_tvm)
        print("✓ TVM kernel执行成功")
    except Exception as e:
        print(f"✗ TVM kernel执行失败: {e}")
        return
    
    # 运行PyTorch实现
    print("\n运行PyTorch实现...")
    try:
        Ai_pytorch = tl_merge_16x16_to_64x64_inverse_pytorch(A, Ad)
        print("✓ PyTorch实现执行成功")
    except Exception as e:
        print(f"✗ PyTorch实现执行失败: {e}")
        return
    
    # 计算误差
    print("\n计算误差...")
    errors = compute_relative_errors(Ai_tvm, Ai_pytorch)
    
    print(f"\n误差统计:")
    print(f"  平均相对误差: {errors['mean_relative_error']:.6e}")
    print(f"  最大相对误差: {errors['max_relative_error']:.6e}")
    print(f"  平均绝对误差: {errors['mean_absolute_error']:.6e}")
    print(f"  最大绝对误差: {errors['max_absolute_error']:.6e}")
    
    # 判断结果是否接近
    tolerance = 1e-3 if dtype == torch.float32 else 1e-2
    if errors['mean_relative_error'] < tolerance:
        print(f"\n✓ 结果匹配！(平均相对误差 < {tolerance})")
    else:
        print(f"\n✗ 结果不匹配！(平均相对误差 >= {tolerance})")
    
    # 性能测试
    if run_benchmark and device == 'cuda':
        print(f"\n{'='*60}")
        print("性能测试")
        print(f"{'='*60}")
        
        print("\nTVM kernel性能测试...")
        Ai_tvm_bench = torch.zeros(B, Token, H, 64, device=device, dtype=dtype)
        tvm_time, tvm_std = benchmark_function(tvm_kernel, A, Ad, Ai_tvm_bench)
        print(f"  平均时间: {tvm_time:.3f} ± {tvm_std:.3f} ms")
        
        print("\nPyTorch实现性能测试...")
        pytorch_time, pytorch_std = benchmark_function(tl_merge_16x16_to_64x64_inverse_pytorch, A, Ad)
        print(f"  平均时间: {pytorch_time:.3f} ± {pytorch_std:.3f} ms")
        
        speedup = pytorch_time / tvm_time
        print(f"\n加速比: {speedup:.2f}x (TVM相比PyTorch)")
    
    print(f"\n{'='*60}\n")


def run_multiple_tests():
    """
    运行多组测试
    """
    print("\n" + "="*60)
    print("块状矩阵逆运算：TVM vs PyTorch 对比测试")
    print("="*60)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU运行")
        device = 'cpu'
    else:
        device = 'cuda'
        print(f"使用设备: {torch.cuda.get_device_name(0)}")
    
    # 测试配置列表
    test_configs = [
        # (B, Token, H, BT)
        # (1, 64, 1, 48),      # 小batch
        # (2, 128, 4, 48),     # 中batch
        (4, 256, 8, 48),     # 大batch
        (8, 512, 16, 48),    # 超大batch
    ]
    
    for B, Token, H, BT in test_configs:
        try:
            compare_implementations(
                B=B, Token=Token, H=H, BT=BT,
                device=device, dtype=torch.float32,
                run_benchmark=True
            )
        except Exception as e:
            print(f"测试失败 (B={B}, Token={Token}, H={H}, BT={BT}): {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    # 运行所有测试
    run_multiple_tests()
    
    # 也可以单独运行一个测试
    # compare_implementations(B=1, Token=64, H=1, BT=48, device='cuda', dtype=torch.float32)

