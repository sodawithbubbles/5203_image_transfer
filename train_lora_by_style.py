import os
import subprocess
from pathlib import Path

# ======= 基础参数 =======
base_model = "runwayml/stable-diffusion-v1-5"
base_dir = Path("style_data")
output_root = Path("lora_outputs")

# ======= 可调超参数 =======
resolution = 512
lr = 5e-4
steps = 1200
batch_size = 1
accum_steps = 4
rank = 4
checkpoint_steps = 200  # 每200步保存一次checkpoint并验证
validation_epochs = 50  # 禁止频繁验证，仅当保存checkpoint时执行

# ======= Wandb 设置 =======
os.environ["WANDB_PROJECT"] = "multi-style-lora"
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "false"

# ======= 要排除的风格 =======
exclude_styles = {"Flatillustration", "Oilpainting", "Sketch", "Watercolor"}

# ======= 批量训练每个风格 =======
for style_dir in base_dir.iterdir():
    if not style_dir.is_dir():
        continue
    if style_dir.name in exclude_styles:
        print(f"⚠️ 跳过排除风格: {style_dir.name}")
        continue

    # 每个风格内部的实际图片文件夹，例如 style_data/Cartoon/cartoon/
    inner_folders = [p for p in style_dir.iterdir() if p.is_dir()]
    if not inner_folders:
        print(f"⚠️ 跳过 {style_dir}：未找到子文件夹")
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

        # 构建训练命令
        cmd = [
            "accelerate", "launch", "train_text_to_image_lora.py",
            f"--pretrained_model_name_or_path={base_model}",
            f"--train_data_dir={inner}",
            "--image_column=image",
            "--caption_column=text",
            f"--validation_prompt=A photo in {style_name} style.",
            "--num_validation_images=4",
            f"--resolution={resolution}",
            f"--train_batch_size={batch_size}",
            f"--gradient_accumulation_steps={accum_steps}",
            f"--learning_rate={lr}",
            f"--max_train_steps={steps}",
            f"--checkpointing_steps={checkpoint_steps}",
            f"--rank={rank}",
            "--mixed_precision=fp16",
            "--lr_scheduler=constant",
            f"--validation_epochs={validation_epochs}",
            "--report_to=wandb",
            f"--output_dir={output_dir}"
        ]

        print(" ".join(cmd))
        subprocess.run(cmd)

print("\n✅ 所有风格训练完成！")