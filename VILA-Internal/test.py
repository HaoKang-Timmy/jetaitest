import fire
import torch
from transformers import AutoConfig
from llava.model import LlavaLlamaModel
from llava.media import Image

def main(model_path="Efficient-Large-Model/0822_jetlm_2b", 
         image_path="./img.png",
         device="cuda"):
    
    prompt = "What is the main subject of the image?"
    
    # Load config
    config = AutoConfig.from_pretrained(model_path)
    config.resume_path = model_path
    
    # Prepare config for eval
    if getattr(config, "vision_tower_cfg", None) is None:
        config.vision_tower_cfg = config.mm_vision_tower
    
    # Prepare kwargs
    kwargs = {"torch_dtype": torch.float16}
    config.model_dtype = kwargs.pop("torch_dtype").__str__()
    
    # Create model (on CPU first due to low_cpu_mem_usage)
    model = LlavaLlamaModel(
        config=config, 
        low_cpu_mem_usage=True,
        **kwargs
    )
    
    # Move model to GPU - this is the KEY step that was missing!
    model.eval().cuda()
    
    # Move vision tower to GPU explicitly
    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        vision_tower.to(device=device, dtype=torch.float16)
    
    # Move mm_projector to GPU explicitly
    mm_projector = model.get_mm_projector()
    if mm_projector is not None:
        mm_projector.to(device=device, dtype=torch.float16)
    
    # Resize token embeddings
    tokenizer = model.tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # Now run inference
    media = Image(image_path)
    output = model.generate_content([media, prompt])
    print(output)
    
if __name__ == "__main__":
    fire.Fire(main)