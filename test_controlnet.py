# controlnet_inference.py
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import cv2
import gc

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from lora_utils import load_lora_weights  # 假设你有加载 LoRA 的函数
from config import BASE_MODEL, OUTPUT_ROOT, BASE_DIR, CONTROLNET_MODEL, \
    GUIDANCE_SCALE, CONTROLNET_SCALE, USE_LORA, EXCLUDE_STYLES

# ===== 工具函数 =====
def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(img, path):
    img.save(path)

def generate_canny(img):
    img_cv = np.array(img)
    edges = cv2.Canny(img_cv, 100, 200)
    return Image.fromarray(edges)

# ===== 主函数 =====
def controlnet_inference(resolution=384, num_steps=50):
    for style_dir in BASE_DIR.iterdir():
        if not style_dir.is_dir() or style_dir.name in EXCLUDE_STYLES:
            print(f"⚠️ 跳过风格: {style_dir.name}")
            continue

        inner_folders = [p for p in style_dir.iterdir() if p.is_dir()]
        if not inner_folders:
            continue

        for inner in inner_folders:
            style_name = style_dir.name
            output_dir = OUTPUT_ROOT / style_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n🎨 处理风格: {style_name}")

            # ===== 初始化 ControlNet pipeline =====
            controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, dtype=torch.float16)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                BASE_MODEL,
                controlnet=controlnet,
                dtype=torch.float16
            )
            pipe.to("cuda")
            pipe.enable_attention_slicing()  # 显存优化

            # ===== 加载 LoRA 权重（可选） =====
            if USE_LORA:
                lora_checkpoint = output_dir / "checkpoint.safetensors"
                if lora_checkpoint.exists():
                    load_lora_weights(pipe.unet, lora_checkpoint, alpha=1.0)

            # ===== 遍历图片逐张生成 =====
            for img_file in inner.iterdir():
                if img_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                    continue

                img = load_image(img_file).resize((resolution, resolution))
                control_image = generate_canny(img)
                prompt = f"A photo in {style_name} style"

                # ===== 推理 =====
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    output_img = pipe(
                        prompt=prompt,
                        image=control_image,
                        guidance_scale=GUIDANCE_SCALE,
                        controlnet_conditioning_scale=CONTROLNET_SCALE,
                        num_inference_steps=num_steps
                    ).images[0]

                save_path = output_dir / f"{img_file.stem}_final.png"
                save_image(output_img, save_path)
                print(f"✅ 生成完成: {save_path}")

                # ===== 显存管理 =====
                del output_img, control_image, img
                torch.cuda.empty_cache()
                gc.collect()

            # ===== 清理 pipeline =====
            del pipe, controlnet
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    controlnet_inference()

