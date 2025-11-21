from PIL import Image
import os

os.makedirs("style_data_512", exist_ok=True)

for f in os.listdir("style_data/data-1000"):
    if f.endswith((".jpg", ".png")):
        img = Image.open(f"style_data/data-1000/{f}").convert("RGB")
        img = img.resize((512, 512))
        img.save(f"style_data_512/{f}")
