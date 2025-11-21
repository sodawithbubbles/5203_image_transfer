import os
import pandas as pd
import json
from pathlib import Path

base_dir = Path("style_data")

for style_dir in base_dir.rglob("train.csv"):
    folder = style_dir.parent
    csv_path = folder / "train.csv"
    print(f"📝 Converting {csv_path} -> metadata.jsonl")

    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 尝试识别列名
    image_col = None
    text_col = None
    for col in df.columns:
        if "image" in col.lower() or "file" in col.lower():
            image_col = col
        if "prompt" in col.lower() or "caption" in col.lower() or "text" in col.lower():
            text_col = col

    if image_col is None or text_col is None:
        print(f"⚠️  Skipping {csv_path}, columns not found.")
        continue

    # 输出路径
    metadata_path = folder / "metadata.jsonl"

    with open(metadata_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            file_name = os.path.basename(str(row[image_col]))
            caption = str(row[text_col]).strip()
            if not caption:
                caption = " "  # 空文本占位
            json_line = {"file_name": file_name, "text": caption}
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"✅ Saved: {metadata_path}")
