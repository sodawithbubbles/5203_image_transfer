# 用同样prompt生成两张图进行对比
from diffusers import StableDiffusionPipeline
import torch

base = "runwayml/stable-diffusion-v1-5"
lora = "lora_outputs/Sketch/pytorch_lora_weights.safetensors"

pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16).to("cuda")

prompt = "An apple in sketch style"
image_base = pipe(prompt).images[0]
image_base.save("base.png")

pipe.load_lora_weights(lora)
image_lora = pipe(prompt).images[0]
image_lora.save("lora.png")
