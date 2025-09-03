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
import torch
from typing import Optional, Union, Dict, List
from transformers import AutoTokenizer, PreTrainedTokenizer

# 导入本地模型模块
from models import JetNemotronForCausalLM, JetNemotronConfig

class JetNemotronWrapper:
    """Jet Nemotron模型的包装类，提供简单的接口来加载和使用模型"""
    
    def __init__(
        self,
        model_name_or_path: str = "jet-ai/Jet-Nemotron-2B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2"
    ):
        """
        初始化JetNemotron模型包装器
        
        Args:
            model_name_or_path: 模型名称或路径
            device: 运行设备 ('cuda' 或 'cpu')
            torch_dtype: 模型权重数据类型
            attn_implementation: 注意力实现方式
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        # 初始化配置
        self.config = JetNemotronConfig.from_pretrained(model_name_or_path)
        self.config.attn_implementation = attn_implementation
        
        # 初始化模型
        self.model = JetNemotronForCausalLM(self.config)
        
        # 加载预训练权重
        state_dict = torch.load(
            os.path.join(model_name_or_path, "model.safetensors")
            if os.path.isdir(model_name_or_path)
            else model_name_or_path
        )
        self.model.load_state_dict(state_dict)
        
        # 设置模型参数
        self.model = self.model.eval()
        self.model = self.model.to(device=device, dtype=torch_dtype)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> str:
        """
        生成文本

        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成token数
            do_sample: 是否使用采样
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: top-p采样参数
            repetition_penalty: 重复惩罚系数
            **kwargs: 其他生成参数

        Returns:
            生成的文本
        """
        # 准备输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "use_cache": True,
            **kwargs
        }
        
        # 生成文本
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)
            
        # 解码输出
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """便捷调用接口"""
        return self.generate(prompt, **kwargs)
