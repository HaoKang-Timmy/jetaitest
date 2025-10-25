"""
调试脚本：找出为什么注释第一个函数后第二个函数会失败
"""
import torch
import sys
sys.path.insert(0, '/storage/home/hcoda1/6/hkang342/p-tkrishna3-0/jetaitest/flash-linear-attention')

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
sys.path.insert(0, '/storage/home/hcoda1/6/hkang342/p-tkrishna3-0/jetaitest/test/edge_models/jetinfra')
from tl_chunked_deltarule_new import tilelang_chunk_gated_delta_rule

print("=" * 80)
print("测试 1: 分别运行两个函数，看输出是否正确")
print("=" * 80)

# 准备测试数据
B, Token, H, K, V = 1, 128, 2, 64, 64
input_dtype = torch.bfloat16
element_dtype = torch.float32

k = torch.randn(B, Token, H, K).to(input_dtype).to("cuda")
w = torch.randn(B, Token, H, K).to(input_dtype).to("cuda")
u = torch.randn(B, Token, H, V).to(input_dtype).to("cuda")
g = torch.randn(B, Token, H).to(element_dtype).to("cuda")

# 保存副本
k1 = k.clone()
w1 = w.clone()
u1 = u.clone()
g1 = g.clone()

k2 = k.clone()
w2 = w.clone()
u2 = u.clone()
g2 = g.clone()

print("\n第一次测试：只运行 Triton 函数")
print("-" * 80)
h_triton, v_new_triton, final_state_triton = chunk_gated_delta_rule_fwd_h(
    k=k1,
    w=w1,
    u=u1,
    g=g1,
    initial_state=None,
    output_final_state=True,
    cu_seqlens=None,
)
print(f"Triton - h stats: mean={h_triton.mean().item():.6e}, std={h_triton.std().item():.6e}")
print(f"Triton - v_new stats: mean={v_new_triton.mean().item():.6e}, std={v_new_triton.std().item():.6e}")
print(f"Triton - final_state stats: mean={final_state_triton.mean().item():.6e}, std={final_state_triton.std().item():.6e}")
print(f"Triton - h包含NaN: {torch.isnan(h_triton).any().item()}, 包含Inf: {torch.isinf(h_triton).any().item()}")

print("\n第二次测试：只运行 TileLang 函数")
print("-" * 80)
h_tilelang, v_new_tilelang, final_state_tilelang = tilelang_chunk_gated_delta_rule(
    batch_size=B,
    k=k2,
    w=w2,
    u=u2,
    g=g2,
    output_final_state=True,
    chunk_size=64,
    save_new_value=True
)
print(f"TileLang - h stats: mean={h_tilelang.mean().item():.6e}, std={h_tilelang.std().item():.6e}")
print(f"TileLang - v_new stats: mean={v_new_tilelang.mean().item():.6e}, std={v_new_tilelang.std().item():.6e}")
print(f"TileLang - final_state stats: mean={final_state_tilelang.mean().item():.6e}, std={final_state_tilelang.std().item():.6e}")
print(f"TileLang - h包含NaN: {torch.isnan(h_tilelang).any().item()}, 包含Inf: {torch.isinf(h_tilelang).any().item()}")

print("\n" + "=" * 80)
print("测试 2: 先运行 Triton 再运行 TileLang")
print("=" * 80)

k3 = k.clone()
w3 = w.clone()
u3 = u.clone()
g3 = g.clone()

k4 = k.clone()
w4 = w.clone()
u4 = u.clone()
g4 = g.clone()

# 先运行 Triton
h_triton2, v_new_triton2, final_state_triton2 = chunk_gated_delta_rule_fwd_h(
    k=k3,
    w=w3,
    u=u3,
    g=g3,
    initial_state=None,
    output_final_state=True,
    cu_seqlens=None,
)
print("已运行 Triton 函数")

# 再运行 TileLang
h_tilelang2, v_new_tilelang2, final_state_tilelang2 = tilelang_chunk_gated_delta_rule(
    batch_size=B,
    k=k4,
    w=w4,
    u=u4,
    g=g4,
    output_final_state=True,
    chunk_size=64,
    save_new_value=True
)
print("已运行 TileLang 函数")

print(f"\nTileLang (在Triton之后) - h stats: mean={h_tilelang2.mean().item():.6e}, std={h_tilelang2.std().item():.6e}")
print(f"TileLang (在Triton之后) - h包含NaN: {torch.isnan(h_tilelang2).any().item()}, 包含Inf: {torch.isinf(h_tilelang2).any().item()}")

print("\n" + "=" * 80)
print("对比结果")
print("=" * 80)

print(f"\n单独运行 vs 先运行Triton后运行 TileLang:")
h_diff = torch.abs(h_tilelang - h_tilelang2).mean().item()
print(f"h 差异: {h_diff:.6e}")

if h_diff < 1e-3:
    print("✓ 两次TileLang运行结果一致！问题可能在其他地方")
else:
    print("✗ 两次TileLang运行结果不一致！这证实了问题的存在")

print(f"\nTileLang vs Triton (在同一测试中):")
h_diff2 = torch.abs(h_tilelang2 - h_triton2).mean().item()
print(f"h 差异: {h_diff2:.6e}")

print("\n" + "=" * 80)
print("测试 3: 检查CUDA上下文和内存")
print("=" * 80)
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"当前设备: {torch.cuda.current_device()}")
print(f"已分配内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"缓存内存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

