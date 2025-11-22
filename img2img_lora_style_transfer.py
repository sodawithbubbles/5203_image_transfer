import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# ======= 路径设置 =======
base_model = "runwayml/stable-diffusion-v1-5"
lora_path = "lora_outputs/Oilpainting/checkpoint-1200"  # 换成你的风格 LoRA 路径
input_image_path = "input_img/1.jpg"                   # 要转换的图像
output_image_path = "output_style.jpg"

# ======= 1. 加载 img2img pipeline =======
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
).to("cuda")

# ======= 2. 加载 LoRA =======
pipe.unet.load_attn_procs(lora_path)

# ======= 3. 加载输入图像 =======
init_image = Image.open(input_image_path).convert("RGB")
init_image = init_image.resize((512, 512))  # SD1.5 推荐 512x512

# ======= 4. 设置风格迁移强度 =======
strength = 0.6
# 0.0 → 几乎完全原图  
# 1.0 → 完全忽略原图，只看风格  
# 0.6~0.8 通常最好

# ======= 5. 推理 =======
prompt = "in oilpainting style"  # 描述风格
result = pipe(
    prompt=prompt,
    image=init_image,
    strength=strength,
    guidance_scale=7.5,
    num_inference_steps=30,
).images[0]

result.save(output_image_path)

print("风格迁移完成！保存到:", output_image_path)
