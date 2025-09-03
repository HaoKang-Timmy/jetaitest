# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from models import JetNemotronForCausalLM, JetNemotronConfig

def get_gpu_memory():
    """获取当前GPU内存使用情况（以GB为单位）"""
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
    current_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
    return current_memory, max_memory

def main():
    parser = argparse.ArgumentParser(description="Benchmark Jet-Nemotron with random tokens")
    parser.add_argument("--model_path", type=str, default="jet-ai/Jet-Nemotron-2B", 
                       help="Path to the model weights")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for generation")
    parser.add_argument("--input_len", type=int, default=4000,
                       help="Input sequence length")
    parser.add_argument("--max_new_tokens", type=int, default=1,
                       help="Number of tokens to generate")
    parser.add_argument("--num_runs", type=int, default=1,
                       help="Number of runs for averaging performance")
    parser.add_argument("--warmup_runs", type=int, default=1,
                       help="Number of warmup runs before timing")
    args = parser.parse_args()

    # 初始化tokenizer和配置
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    config = JetNemotronConfig.from_pretrained(args.model_path)
    config.attn_implementation = "flash_attention_2"
    
    # 初始化模型
    model = JetNemotronForCausalLM(config)
    model = model.eval().cuda().bfloat16()
    
    # 加载模型权重
    if os.path.isfile(args.model_path) or os.path.isdir(args.model_path):
        if os.path.isfile(args.model_path):
            state_dict = torch.load(args.model_path)
        else:
            state_dict = torch.load(os.path.join(args.model_path, "model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        print(f"Loading weights from HuggingFace model: {args.model_path}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        model.load_state_dict(hf_model.state_dict())
        del hf_model

    torch.cuda.reset_peak_memory_stats()

    # 生成随机输入tokens
    vocab_size = model.config.vocab_size
    random_tokens = torch.randint(
        0, vocab_size, 
        (args.batch_size, args.input_len), 
        device="cuda"
    )

    # 生成参数
    gen_kwargs = {
        "do_sample": False,
        "use_cache": True,
        "attention_mask": torch.ones_like(random_tokens)
    }

    # 预热运行
    print(f"\nRunning {args.warmup_runs} warmup iterations...")
    for _ in range(args.warmup_runs):
        with torch.no_grad():
            _ = model.generate(random_tokens, max_new_tokens=1, **gen_kwargs)

    # 计时运行
    print(f"\nRunning {args.num_runs} timed iterations...")
    
    prefill_times = []
    total_times = []

    for _ in range(args.num_runs):
        torch.cuda.synchronize()
        
        # 测量prefill时间（生成一个token）
        start_time = time.time()
        with torch.no_grad():
            _ = model.generate(random_tokens, max_new_tokens=1, **gen_kwargs)
        torch.cuda.synchronize()
        prefill_time = time.time() - start_time
        prefill_times.append(prefill_time)
        
        # 测量总时间（生成指定数量的token）
        start_time = time.time()
        with torch.no_grad():
            _ = model.generate(random_tokens, max_new_tokens=args.max_new_tokens, **gen_kwargs)
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        total_times.append(total_time)

    # 获取GPU内存使用情况
    current_mem, peak_mem = get_gpu_memory()

    # 计算平均时间
    avg_prefill = sum(prefill_times) / args.num_runs
    avg_total = sum(total_times) / args.num_runs
    avg_decode = (avg_total - avg_prefill) / (args.max_new_tokens - 1) if args.max_new_tokens > 1 else 0
    
    print("\nPerformance Summary:")
    print(f"Prefill time: {avg_prefill*1000:.3f}ms")
    print(f"Decode time per token: {avg_decode*1000:.3f}ms")
    
    print(f"\nGPU Memory Usage:")
    print(f"Peak memory: {peak_mem:.2f}GB")
    print(f"Current memory: {current_mem:.2f}GB")
    
    print("\nConfiguration:")
    print(f"Batch size: {args.batch_size}")
    print(f"Input sequence length: {args.input_len}")
    print(f"Generated tokens: {args.max_new_tokens}")
    print(f"Model: {args.model_path}")

if __name__ == "__main__":
    main()