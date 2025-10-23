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

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from server_models import JetNemotronForCausalLM, JetNemotronConfig

parser = argparse.ArgumentParser(description="Generate text using Jet-Nemotron")
parser.add_argument("--model_name_or_path", type=str, default="jet-ai/Jet-Nemotron-2B", 
                   help="Path to the model directory")
parser.add_argument("--input_text", type=str, default="Hello, I'm Jet-Nemotron from NVIDIA.",
                   help="Input text for generation")
parser.add_argument("--max_new_tokens", type=int, default=50,
                   help="Maximum number of new tokens to generate")
args = parser.parse_args()

model_name_or_path = args.model_name_or_path

# Initialize config and model using server_models
config = JetNemotronConfig.from_pretrained(model_name_or_path)
config.attn_implementation = "flash_attention_2"

model = JetNemotronForCausalLM(config)
model = model.to("cuda").bfloat16().eval()

# Load weights from HuggingFace model
print(f"Loading weights from: {model_name_or_path}")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model.load_state_dict(hf_model.state_dict())
del hf_model  # Release memory

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

# Generate
input_ids = tokenizer(args.input_text, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
output_str = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_str)