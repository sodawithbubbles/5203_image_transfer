import os
import json

# 配置参数
DATA_DIR = "style_data_512"  # 你的图片目录
PROMPT = "a photo in anime style"  # 你的提示词
OUTPUT_FILE = "metadata.jsonl"  # 输出文件名

# 创建 metadata.jsonl
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
image_files = [f for f in os.listdir(DATA_DIR) 
               if any(f.lower().endswith(ext) for ext in image_extensions)]

metadata_path = os.path.join(DATA_DIR, OUTPUT_FILE)
with open(metadata_path, 'w', encoding='utf-8') as f:
    for image_file in image_files:
        metadata = {"file_name": image_file, "text": PROMPT}
        f.write(json.dumps(metadata) + '\n')

print(f"成功创建 {metadata_path}")
print(f"处理了 {len(image_files)} 张图片")