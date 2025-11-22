from pathlib import Path

# 基础模型
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# 数据集路径
BASE_DIR = Path("style_data")
OUTPUT_ROOT = Path("pipeline_outputs")

# LoRA训练参数
RESOLUTION = 384
LR = 5e-4
STEPS = 1200
BATCH_SIZE = 1
ACCUM_STEPS = 4
RANK = 4
CHECKPOINT_STEPS = 200
VALIDATION_EPOCHS = 50

# ControlNet
CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"
GUIDANCE_SCALE = 7.5
CONTROLNET_SCALE = 1.0

USE_LORA = True
# InstantStyle
USE_INSTANTSTYLE = False
INSTANTSTYLE_CHECKPOINT = "instantstyle-plus.ckpt"

# 排除风格
EXCLUDE_STYLES = {"Flatillustration", "Oilpainting", "Sketch", "Watercolor"}
