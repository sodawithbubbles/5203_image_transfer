import os
import subprocess
from pathlib import Path

# ======= 基础参数 =======
base_model = "runwayml/stable-diffusion-v1-5"
base_dir = Path("style_data")
output_root = Path("lora_outputs")

# 可调参数
resolution = 512
lr = 5e-4
steps = 1200
batch_size = 1
accum_steps = 4
rank = 4

# ======= 批量训练每个风格 =======
for style_dir in base_dir.iterdir():
    if not style_dir.is_dir():
        continue

    # 每个风格内部的实际图片文件夹，例如 style_data/Anime/anime/
    inner_folders = [p for p in style_dir.iterdir() if p.is_dir()]
    if not inner_folders:
        continue

    for inner in inner_folders:
        metadata_file = inner / "metadata.jsonl"
        if not metadata_file.exists():
            print(f"⚠️ 跳过 {inner}：未找到 metadata.jsonl")
            continue

        style_name = style_dir.name
        output_dir = output_root / style_name
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🚀 开始训练风格: {style_name}")
        cmd = [
            "accelerate", "launch", "train_text_to_image_lora.py",
            f"--pretrained_model_name_or_path={base_model}",
            f"--train_data_dir={inner}",
            "--image_column=image",
            "--caption_column=text",
            f"--resolution={resolution}",
            f"--train_batch_size={batch_size}",
            f"--gradient_accumulation_steps={accum_steps}",
            f"--learning_rate={lr}",
            f"--max_train_steps={steps}",
            "--checkpointing_steps=200",
            "--lr_scheduler=constant",
            "--report_to"
            f"--rank={rank}",
            f"--output_dir={output_dir}",
            "--validation_epochs=50"
        ]

        print(" ".join(cmd))
        subprocess.run(cmd)