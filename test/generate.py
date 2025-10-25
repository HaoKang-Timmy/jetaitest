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
import argparse
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from edge_models import JetNemotronForCausalLM, JetNemotronConfig

def main():
    parser = argparse.ArgumentParser(description="Generate text using local Jet-Nemotron")
    parser.add_argument("--model_path", type=str, default="jet-ai/Jet-Nemotron-2B", 
                       help="Path to the model weights")
    # parser.add_argument("--input_text", type=str, default="Hi, My name is Timmy. I am currently a 3rd year PhD student in Computer Science at Stanford University. I am interested in natural language processing and machine learning. How about you? Do you like to travel? Or do you like to read books? What is your favorite book? What is your favorite travel destination? Please",
    #                    help="Input text for generation")
    parser.add_argument("--input_text", type=str, default="Hi, My name is Timmy. I am currently a 3rd year PhD student in Computer Science at Stanford University. I am interested in natural language processing and machine learning. How about you? Do you like to travel? Or do you like to read books? What is your favorite book? What is your favorite travel destination? Hi, My name is Timmy. I am currently a 3rd year PhD student in Computer Science at Stanford University. I am interested in natural language processing and machine learning. How about you? Do you like to travel? Or do you like to read books? What is your favorite book? What is your favorite travel destination? Please",
                       help="Input text for generation")
    # parser.add_argument("--input_text", type=str, default="Hi, My name is Timmy.",
                    #    help="Input text for generation")
    parser.add_argument("--max_new_tokens", type=int, default=30,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--do_sample", action="store_true",
                       help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=None,
                       help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty coefficient")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # 设置随机种子以确保可复现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # 设置确定性行为（可能会影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 禁用torch.compile和CUDA graphs
    # torch._dynamo.config.suppress_errors = True
    # torch._dynamo.config.disable = True
    # os.environ["TORCH_COMPILE_DISABLE"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 禁用异步执行，间接禁用CUDA graphs

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 初始化配置
    config = JetNemotronConfig.from_pretrained(args.model_path)
    config.attn_implementation = "flash_attention_2"
    
    # 初始化模型
    model = JetNemotronForCausalLM(config)
    model = model.eval().to("cuda").bfloat16()
    
    # 加载模型权重
    if os.path.isfile(args.model_path) or os.path.isdir(args.model_path):
        # 如果是本地文件或目录，直接加载
        if os.path.isfile(args.model_path):
            state_dict = torch.load(args.model_path)
        else:
            state_dict = torch.load(os.path.join(args.model_path, "model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        # 如果是HuggingFace模型ID，使用from_pretrained加载
        print(f"Loading weights from HuggingFace model: {args.model_path}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        model.load_state_dict(hf_model.state_dict())
        del hf_model  # 释放内存

    # 准备输入
    inputs = tokenizer(args.input_text, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")

    # 生成参数
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "use_cache": True
    }

    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)
        
    # 解码输出
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 打印结果
    print("\nInput:", args.input_text)
    print("\nOutput:", output_text)

if __name__ == "__main__":
    main()