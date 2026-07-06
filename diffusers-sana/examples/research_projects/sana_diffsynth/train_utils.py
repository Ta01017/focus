import json
from pathlib import Path

import torch
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file

from diffusers import SanaPipeline
from diffusers.utils import convert_unet_state_dict_to_peft


DEFAULT_LORA_TARGETS = "to_q,to_k,to_v,to_out.0"
COMPLEX_HUMAN_INSTRUCTION = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for "
    "image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial "
    "relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping "
    "peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street "
    "lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass "
    "skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:",
    "User Prompt: ",
]


def add_lora(transformer, rank, alpha, target_modules, gradient_checkpointing):
    transformer.requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=[name.strip() for name in target_modules.split(",") if name.strip()],
        )
    )
    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()


def save_training_weights(accelerator, model, output_dir, config, global_step):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    SanaPipeline.save_lora_weights(
        output_dir,
        transformer_lora_layers=get_peft_model_state_dict(unwrapped.transformer),
    )
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in unwrapped.adapter.state_dict().items()},
        output_dir / "condition_adapter.safetensors",
    )
    config = {**config, "global_step": global_step}
    (output_dir / "condition_adapter.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_adapter_weights(adapter, checkpoint):
    path = Path(checkpoint) / "condition_adapter.safetensors"
    adapter.load_state_dict(load_file(path), strict=True)


def load_lora_weights(transformer, checkpoint):
    state_dict = SanaPipeline.lora_state_dict(checkpoint)
    state_dict = {
        key.removeprefix("transformer."): value
        for key, value in state_dict.items()
        if key.startswith("transformer.")
    }
    state_dict = convert_unet_state_dict_to_peft(state_dict)
    incompatible = set_peft_model_state_dict(transformer, state_dict, adapter_name="default")
    if incompatible is not None and getattr(incompatible, "unexpected_keys", None):
        raise ValueError(f"Unexpected LoRA keys in {checkpoint}: {incompatible.unexpected_keys}")


def trainable_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def get_weight_dtype(mixed_precision):
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32
