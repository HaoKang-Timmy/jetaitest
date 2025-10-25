"""
调试 NaN 问题：为什么单独运行 TileLang kernel 会产生 NaN
但先运行其他函数就不会出现 NaN
"""
import torch
import sys
sys.path.insert(0, '/storage/home/hcoda1/6/hkang342/p-tkrishna3-0/jetaitest/flash-linear-attention')
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h

sys.path.insert(0, '/storage/home/hcoda1/6/hkang342/p-tkrishna3-0/jetaitest/test/edge_models/jetinfra')
from tl_chunked_deltarule_new import tilelang_chunk_gated_delta_rule

print("=" * 80)
print("调试 NaN 问题")
print("=" * 80)

# 准备测试数据
B, Token, H, K, V = 1, 256, 4, 64, 64
input_dtype = torch.bfloat16
element_dtype = torch.float32

def create_test_data():
    """创建新的测试数据"""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    k = torch.randn(B, Token, H, K).to(input_dtype).to("cuda")
    w = torch.randn(B, Token, H, K).to(input_dtype).to("cuda")
    u = torch.randn(B, Token, H, V).to(input_dtype).to("cuda")
    g = torch.randn(B, Token, H).to(element_dtype).to("cuda")
    return k, w, u, g

def check_nan_inf(tensor, name):
    """检查张量是否包含 NaN 或 Inf"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        print(f"  ✗ {name}: NaN={has_nan}, Inf={has_inf}")
        if has_nan:
            nan_count = torch.isnan(tensor).sum().item()
            total = tensor.numel()
            print(f"    NaN元素: {nan_count}/{total} ({100*nan_count/total:.2f}%)")
        if has_inf:
            inf_count = torch.isinf(tensor).sum().item()
            total = tensor.numel()
            print(f"    Inf元素: {inf_count}/{total} ({100*inf_count/total:.2f}%)")
        return False
    else:
        print(f"  ✓ {name}: 健康")
        return True

# ============================================================================
print("\n" + "=" * 80)
print("测试 1: 直接运行 TileLang kernel (全新CUDA环境)")
print("=" * 80)

# 重启 CUDA 上下文
torch.cuda.empty_cache()
torch.cuda.synchronize()

k, w, u, g = create_test_data()

print("\n输入检查:")
all_input_ok = True
all_input_ok &= check_nan_inf(k, "k")
all_input_ok &= check_nan_inf(w, "w")
all_input_ok &= check_nan_inf(u, "u")
all_input_ok &= check_nan_inf(g, "g")

if not all_input_ok:
    print("\n✗ 输入数据有问题，测试终止")
    sys.exit(1)

print("\n运行 TileLang kernel...")
h_tl, v_new_tl, final_state_tl = tilelang_chunk_gated_delta_rule(
    batch_size=B, k=k, w=w, u=u, g=g,
    output_final_state=True, chunk_size=64, save_new_value=True
)
torch.cuda.synchronize()

print("\n输出检查:")
tl_direct_ok = True
tl_direct_ok &= check_nan_inf(h_tl, "h")
tl_direct_ok &= check_nan_inf(v_new_tl, "v_new")
tl_direct_ok &= check_nan_inf(final_state_tl, "final_state")

if not tl_direct_ok:
    print("\n✗✗✗ 确认问题: 直接运行 TileLang 产生 NaN/Inf")
else:
    print("\n✓ 直接运行 TileLang 正常")

# ============================================================================
print("\n" + "=" * 80)
print("测试 2: 先运行 Triton，再运行 TileLang (相同输入)")
print("=" * 80)

# 重新创建相同的数据
k2 = k.clone()
w2 = w.clone()
u2 = u.clone()
g2 = g.clone()

print("\n先运行 Triton kernel...")
h_triton, v_new_triton, final_state_triton = chunk_gated_delta_rule_fwd_h(
    k=k2, w=w2, u=u2, g=g2,
    initial_state=None, output_final_state=True, cu_seqlens=None
)
torch.cuda.synchronize()

print("Triton 输出检查:")
triton_ok = True
triton_ok &= check_nan_inf(h_triton, "h_triton")
triton_ok &= check_nan_inf(v_new_triton, "v_new_triton")
triton_ok &= check_nan_inf(final_state_triton, "final_state_triton")

print("\n再运行 TileLang kernel (相同输入)...")
h_tl2, v_new_tl2, final_state_tl2 = tilelang_chunk_gated_delta_rule(
    batch_size=B, k=k, w=w, u=u, g=g,
    output_final_state=True, chunk_size=64, save_new_value=True
)
torch.cuda.synchronize()

print("\nTileLang 输出检查 (在 Triton 之后):")
tl_after_triton_ok = True
tl_after_triton_ok &= check_nan_inf(h_tl2, "h")
tl_after_triton_ok &= check_nan_inf(v_new_tl2, "v_new")
tl_after_triton_ok &= check_nan_inf(final_state_tl2, "final_state")

if tl_after_triton_ok and not tl_direct_ok:
    print("\n✓✓✓ 确认: 先运行 Triton 可以'修复' TileLang 的 NaN 问题")
else:
    print("\n结果与预期不符")

# ============================================================================
print("\n" + "=" * 80)
print("测试 3: 只运行一个简单的 CUDA 预热操作")
print("=" * 80)

# 完全清空 CUDA 缓存
torch.cuda.empty_cache()
torch.cuda.synchronize()

# 创建新数据
k3, w3, u3, g3 = create_test_data()

print("\n运行简单的 CUDA 预热操作...")
# 简单的 CUDA 操作来初始化上下文
dummy = torch.zeros(1024, 1024, device='cuda', dtype=torch.float32)
dummy = dummy + 1.0
torch.cuda.synchronize()
del dummy

print("预热完成，运行 TileLang kernel...")
h_tl3, v_new_tl3, final_state_tl3 = tilelang_chunk_gated_delta_rule(
    batch_size=B, k=k3, w=w3, u=u3, g=g3,
    output_final_state=True, chunk_size=64, save_new_value=True
)
torch.cuda.synchronize()

print("\nTileLang 输出检查 (简单预热后):")
tl_after_warmup_ok = True
tl_after_warmup_ok &= check_nan_inf(h_tl3, "h")
tl_after_warmup_ok &= check_nan_inf(v_new_tl3, "v_new")
tl_after_warmup_ok &= check_nan_inf(final_state_tl3, "final_state")

if tl_after_warmup_ok:
    print("\n✓ 简单预热也能'修复'问题")
else:
    print("\n✗ 简单预热不能修复问题")

# ============================================================================
print("\n" + "=" * 80)
print("测试 4: 运行 TileLang kernel 两次 (第二次是否还有问题)")
print("=" * 80)

# 清空缓存
torch.cuda.empty_cache()
torch.cuda.synchronize()

k4, w4, u4, g4 = create_test_data()

print("\n第一次运行 TileLang...")
h_first, v_new_first, final_state_first = tilelang_chunk_gated_delta_rule(
    batch_size=B, k=k4, w=w4, u=u4, g=g4,
    output_final_state=True, chunk_size=64, save_new_value=True
)
torch.cuda.synchronize()

print("第一次输出检查:")
first_ok = True
first_ok &= check_nan_inf(h_first, "h")
first_ok &= check_nan_inf(v_new_first, "v_new")
first_ok &= check_nan_inf(final_state_first, "final_state")

print("\n第二次运行 TileLang (相同输入)...")
h_second, v_new_second, final_state_second = tilelang_chunk_gated_delta_rule(
    batch_size=B, k=k4, w=w4, u=u4, g=g4,
    output_final_state=True, chunk_size=64, save_new_value=True
)
torch.cuda.synchronize()

print("第二次输出检查:")
second_ok = True
second_ok &= check_nan_inf(h_second, "h")
second_ok &= check_nan_inf(v_new_second, "v_new")
second_ok &= check_nan_inf(final_state_second, "final_state")

if second_ok and not first_ok:
    print("\n✓✓✓ 确认: 第一次运行有问题，第二次运行正常")
    print("    这强烈暗示是 kernel 编译/初始化问题")
elif not first_ok and not second_ok:
    print("\n两次都有问题")
else:
    print("\n两次都正常")

# ============================================================================
print("\n" + "=" * 80)
print("最终诊断")
print("=" * 80)

if not tl_direct_ok and tl_after_triton_ok:
    print("\n诊断结果:")
    print("  问题: TileLang kernel 在第一次运行时产生 NaN")
    print("  修复: 先运行其他 CUDA kernel 可以避免此问题")
    print("\n可能原因:")
    print("  1. TileLang kernel 的某些 buffer 未正确初始化 (最可能)")
    print("  2. CUDA 上下文需要完全初始化")
    print("  3. GPU 寄存器/共享内存需要预热")
    print("\n建议检查:")
    print("  - tilelang_chunk_gated_delta_rule_fwd_h kernel 中的所有 T.clear() 调用")
    print("  - 特别是 b_h_fragment 的初始化")
    print("  - V_new_fragment 和相关 fragment 的初始化")
    print("  - 查看是否有任何 fragment 在使用前没有被正确初始化或清零")
elif tl_direct_ok:
    print("\n✓ 无法复现问题 - TileLang kernel 运行正常")
else:
    print("\n✗ 问题复杂 - 需要进一步分析")

print("\n" + "=" * 80)

