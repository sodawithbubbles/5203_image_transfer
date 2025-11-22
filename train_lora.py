# train_lora.py
import os
import subprocess
from pathlib import Path
from config import BASE_MODEL, BASE_DIR, OUTPUT_ROOT, RESOLUTION, LR, STEPS, BATCH_SIZE, ACCUM_STEPS, \
    RANK, CHECKPOINT_STEPS, VALIDATION_EPOCHS, EXCLUDE_STYLES

def train_lora():
    for style_dir in BASE_DIR.iterdir():
        if not style_dir.is_dir() or style_dir.name in EXCLUDE_STYLES:
            print(f"⚠️ 跳过风格: {style_dir.name}")
            continue

        inner_folders = [p for p in style_dir.iterdir() if p.is_dir()]
        if not inner_folders:
            print(f"⚠️ 跳过 {style_dir}: 无子文件夹")
            continue

        for inner in inner_folders:
            metadata_file = inner / "metadata.jsonl"
            if not metadata_file.exists():
                print(f"⚠️ 跳过 {inner}: 无 metadata.jsonl")
                continue

            style_name = style_dir.name
            output_dir = OUTPUT_ROOT / style_name
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n🚀 开始训练风格: {style_name}")

            cmd = [
                "accelerate", "launch", "train_text_to_image_lora.py",
                f"--pretrained_model_name_or_path={BASE_MODEL}",
                f"--train_data_dir={inner}",
                "--image_column=image",
                "--caption_column=text",
                f"--validation_prompt=A photo in {style_name} style.",
                "--num_validation_images=4",
                f"--resolution={RESOLUTION}",
                f"--train_batch_size={BATCH_SIZE}",
                f"--gradient_accumulation_steps={ACCUM_STEPS}",
                f"--learning_rate={LR}",
                f"--max_train_steps={STEPS}",
                f"--checkpointing_steps={CHECKPOINT_STEPS}",
                f"--rank={RANK}",
                "--mixed_precision=fp16",
                "--lr_scheduler=constant",
                f"--validation_epochs={VALIDATION_EPOCHS}",
                "--report_to=wandb",
                f"--output_dir={output_dir}"
            ]

            print(" ".join(cmd))
            subprocess.run(cmd)

    print("\n✅ 所有风格 LoRA 训练完成！")

if __name__ == "__main__":
    train_lora()
