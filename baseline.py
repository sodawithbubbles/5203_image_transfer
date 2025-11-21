# sd v1.5作为baseline生成图片
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype="float16").to("cuda")
img = pipe("a photo of a cat in anime style").images[0]
img.save("baseline.png")