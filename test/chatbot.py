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
import sys
import argparse
import random
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from edge_models import JetNemotronForCausalLM, JetNemotronConfig


class JetChatBot:
    def __init__(self, model_path, seed=42, keep_history=False):
        """Initialize the chatbot"""
        self.model_path = model_path
        self.seed = seed
        self.keep_history = keep_history
        self.conversation_history = []
        
        # Set random seed
        self._set_seed()
        
        # Disable torch.compile and CUDA graphs
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.disable = True
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Load tokenizer and model
        print("Loading model, please wait...")
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        print("Model loaded successfully!\n")
    
    def _set_seed(self):
        """Set random seed for reproducibility"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        return AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
    
    def _load_model(self):
        """Load model"""
        # Initialize config
        config = JetNemotronConfig.from_pretrained(self.model_path)
        config.attn_implementation = "flash_attention_2"
        
        # Initialize model
        model = JetNemotronForCausalLM(config)
        model = model.eval().to("cuda").bfloat16()
        
        # Load model weights
        if os.path.isfile(self.model_path) or os.path.isdir(self.model_path):
            if os.path.isfile(self.model_path):
                state_dict = torch.load(self.model_path)
            else:
                state_dict = torch.load(os.path.join(self.model_path, "model.safetensors"))
            model.load_state_dict(state_dict)
        else:
            # Load from HuggingFace
            print(f"Loading weights from HuggingFace: {self.model_path}")
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cuda"
            )
            model.load_state_dict(hf_model.state_dict())
            del hf_model
        
        return model
    
    def stream_generate(self, input_text, max_new_tokens=100, temperature=0.7, 
                       top_k=50, top_p=0.9, repetition_penalty=1.0):
        """Stream generate text with token-by-token output"""
        # Build full conversation text (if keeping history)
        if self.keep_history and self.conversation_history:
            full_text = "\n".join(self.conversation_history) + "\n" + input_text
        else:
            full_text = input_text
        
        # Prepare input
        inputs = self.tokenizer(full_text, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        
        # Print input
        print(f"\n{'='*60}")
        print(f"You: {input_text}")
        print(f"{'='*60}")
        print("Bot: ", end="", flush=True)
        
        # Create text streamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True if temperature > 0 else False,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "use_cache": True,
            "streamer": streamer
        }
        
        # Start timing
        start_time = time.time()
        
        # Generate text
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Extract generated text (excluding input)
        generated_ids = output_ids[0][input_ids.shape[1]:]
        num_generated_tokens = len(generated_ids)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Calculate tokens per second
        tokens_per_second = num_generated_tokens / elapsed_time if elapsed_time > 0 else 0
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Generated {num_generated_tokens} tokens in {elapsed_time:.2f}s")
        print(f"Speed: {tokens_per_second:.2f} tokens/second")
        print(f"{'='*60}")
        
        # Update conversation history
        if self.keep_history:
            self.conversation_history.append(f"User: {input_text}")
            self.conversation_history.append(f"Bot: {generated_text}")
            # Limit history length to avoid context overflow
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
        
        return output_ids
    
    def chat(self, max_new_tokens=100, temperature=0.7, top_k=50, 
             top_p=0.9, repetition_penalty=1.0):
        """Start chat loop"""
        print("\n" + "="*60)
        print("Jet-Nemotron Chatbot")
        print("Type 'quit', 'exit', or 'q' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Stream generate response
                self.stream_generate(
                    user_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
                
            except KeyboardInterrupt:
                print("\n\nInterrupt detected, exiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description="Jet-Nemotron Chatbot")
    parser.add_argument("--model_path", type=str, default="jet-ai/Jet-Nemotron-2B",
                       help="Path to the model")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                       help="Repetition penalty coefficient")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--keep_history", action="store_true",
                       help="Keep conversation history (experimental feature)")
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = JetChatBot(
        model_path=args.model_path, 
        seed=args.seed,
        keep_history=args.keep_history
    )
    
    # Start chat
    chatbot.chat(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )


if __name__ == "__main__":
    main()

