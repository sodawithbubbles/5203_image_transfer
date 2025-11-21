from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch

base_model = "runwayml/stable-diffusion-v1-5"
lora_path = "lora_outputs/checkpoint-800"

pipe = StableDiffusionPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16
).to("cuda")

# 加载 LoRA
pipe.unet.load_attn_procs(lora_path)

prompt = "a portrait of a cat in anime style"
image = pipe(prompt, num_inference_steps=30).images[0]
image.save("lora_result.png")
