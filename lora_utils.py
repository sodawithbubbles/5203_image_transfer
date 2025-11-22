# lora_utils.py
import torch
from safetensors.torch import load_file as load_safetensors

def load_lora_weights(unet, lora_path, alpha=1.0):
    """
    将 LoRA 权重加载到 Stable Diffusion UNet 模型中
    
    参数：
        unet: diffusers pipeline 的 unet 模块
        lora_path: .safetensors 权重路径
        alpha: LoRA强度（权重缩放），可调节风格影响力
    """
    # 加载 LoRA 权重
    lora_state = load_safetensors(str(lora_path))

    # 遍历 UNet 模型参数，匹配加载 LoRA 权重
    for name, param in unet.named_parameters():
        if name in lora_state:
            lora_tensor = lora_state[name].to(param.device, dtype=param.dtype)
            param.data += alpha * lora_tensor
        else:
            # 某些 LoRA 权重可能只覆盖部分模块，可跳过
            continue

    print(f"✅ LoRA 权重 {lora_path} 已加载，alpha={alpha}")
