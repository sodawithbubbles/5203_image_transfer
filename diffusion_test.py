from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

image = pipe("a photo of a cat in anime style").images[0]
image.save("test.png")
print("✅ Stable Diffusion pipeline works!")
