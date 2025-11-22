import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision import transforms
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# =================== 设置 ===================
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = "runwayml/stable-diffusion-v1-5"
lora_path = "lora_outputs/Anime/checkpoint-1200"
output_dir = "evaluation_comparison"
os.makedirs(output_dir, exist_ok=True)

prompts = [
    "a portrait of a cat in anime style",
    "a scenic landscape in anime style",
    "a cute dog in anime style"
]

ref_dir = "style_data/Anime/anime/"  # 替换为你的参考图
ref_images = sorted([os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".png")])[:len(prompts)]

# =================== 加载模型 ===================
# Baseline
pipe_base = StableDiffusionPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16
).to(device)

# LoRA
pipe_lora = StableDiffusionPipeline.from_pretrained(
    base_model, torch_dtype=torch.float16
).to(device)
pipe_lora.unet.load_attn_procs(lora_path)

# =================== 预处理 ===================
# LPIPS 用 [-1,1] float tensor
lpips_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# FID 用 uint8 tensor
def pil_to_uint8_tensor(img: Image.Image, device='cpu'):
    """
    PIL -> torch.Tensor (C,H,W), dtype=uint8, range [0,255]
    """
    arr = np.array(img)
    tensor = torch.from_numpy(arr).permute(2,0,1).to(device)  # C,H,W
    return tensor

# =================== LPIPS ===================
lpips_fn = lpips.LPIPS(net='alex').to(device)

def compute_lpips(img_gen, img_ref):
    x = lpips_transform(img_gen).unsqueeze(0).to(device)
    y = lpips_transform(img_ref).unsqueeze(0).to(device)
    return lpips_fn(x, y).item()

# =================== FID ===================
fid_base = FrechetInceptionDistance(feature=2048).to(device)
fid_lora = FrechetInceptionDistance(feature=2048).to(device)

# =================== CLIP Score ===================
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def compute_clip_score(img, prompt):
    inputs = clip_processor(text=[prompt], images=img, return_tensors="pt").to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs[0][0].item()

# =================== 生成并评估 ===================
results = []

for idx, prompt in enumerate(prompts):
    ref_img = Image.open(ref_images[idx]).convert("RGB")

    # Baseline
    gen_base = pipe_base(prompt, num_inference_steps=30).images[0]
    lpips_base = compute_lpips(gen_base, ref_img)
    fid_base.update(pil_to_uint8_tensor(gen_base, device=device).unsqueeze(0), real=False)
    fid_base.update(pil_to_uint8_tensor(ref_img, device=device).unsqueeze(0), real=True)
    clip_base = compute_clip_score(gen_base, prompt)
    gen_base.save(os.path.join(output_dir, f"base_{idx}.png"))

    # LoRA
    gen_lora = pipe_lora(prompt, num_inference_steps=30).images[0]
    lpips_lora = compute_lpips(gen_lora, ref_img)
    fid_lora.update(pil_to_uint8_tensor(gen_lora, device=device).unsqueeze(0), real=False)
    fid_lora.update(pil_to_uint8_tensor(ref_img, device=device).unsqueeze(0), real=True)
    clip_lora = compute_clip_score(gen_lora, prompt)
    gen_lora.save(os.path.join(output_dir, f"lora_{idx}.png"))

    # 保存结果
    results.append({
        "prompt": prompt,
        "LPIPS_base": lpips_base,
        "LPIPS_lora": lpips_lora,
        "CLIP_base": clip_base,
        "CLIP_lora": clip_lora
    })

# FID 计算
fid_base_score = fid_base.compute().item()
fid_lora_score = fid_lora.compute().item()

# =================== 输出表格 ===================
df = pd.DataFrame(results)
df["FID_base"] = fid_base_score
df["FID_lora"] = fid_lora_score
df.to_csv(os.path.join(output_dir, "evaluation_comparison.csv"), index=False)
print(df)

# =================== 可视化 ===================
plt.figure(figsize=(10,5))
plt.plot(range(len(prompts)), df["LPIPS_base"], label="LPIPS Baseline", marker='o')
plt.plot(range(len(prompts)), df["LPIPS_lora"], label="LPIPS LoRA", marker='o')
plt.xlabel("Prompt index")
plt.ylabel("LPIPS (Lower better)")
plt.title("LPIPS Comparison")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "lpips_comparison.png"))
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(len(prompts)), df["CLIP_base"], label="CLIP Baseline", marker='o')
plt.plot(range(len(prompts)), df["CLIP_lora"], label="CLIP LoRA", marker='o')
plt.xlabel("Prompt index")
plt.ylabel("CLIP Score (Higher better)")
plt.title("CLIP Score Comparison")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "clip_comparison.png"))
plt.show()

print(f"✅ Evaluation finished! Results saved in {output_dir}")
