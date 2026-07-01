import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def add_metadata_args(parser, metadata_required=False):
    existing = parser._option_string_actions
    arguments = (
        ("--dataset_metadata_path", {"required": metadata_required}),
        ("--dataset_base_path", {"default": "."}),
        ("--target_key", {"default": "image"}),
        ("--edit_key", {"default": "edit_image"}),
        ("--prompt_key", {"default": "prompt"}),
        ("--id_key", {"default": "id"}),
        ("--seed_key", {"default": "seed"}),
        ("--result_key", {"default": "generated_image"}),
        ("--start_index", {"type": int, "default": 0}),
        ("--max_samples", {"type": int, "default": None}),
        ("--skip_existing", {"action": "store_true"}),
        ("--continue_on_error", {"action": "store_true"}),
    )
    for name, kwargs in arguments:
        if name not in existing:
            parser.add_argument(name, **kwargs)
    return parser


def add_pretrained_args(parser):
    existing = parser._option_string_actions
    if "--local_files_only" not in existing:
        parser.add_argument("--local_files_only", action="store_true")
    if "--cache_dir" not in existing:
        parser.add_argument("--cache_dir", default=None)
    if "--revision" not in existing:
        parser.add_argument("--revision", default=None)
    return parser


def pretrained_kwargs(args):
    return {
        "local_files_only": args.local_files_only,
        "cache_dir": args.cache_dir,
        "revision": args.revision,
    }


def select_records(records, start_index, max_samples):
    if start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    if max_samples is not None and max_samples < 1:
        raise ValueError("--max_samples must be positive when provided.")
    end = None if max_samples is None else start_index + max_samples
    return list(enumerate(records[start_index:end], start=start_index))


def sample_output_path(record, index, edit_key, id_key, output_dir):
    edit_images = record[edit_key]
    sample_id = str(record.get(id_key, Path(edit_images[0]).stem))
    safe_id = sample_id.replace("/", "_").replace("\\", "_")
    return Path(output_dir) / f"{index:06d}_{safe_id}.png"


def sample_prompt(record, prompt_key, fallback):
    prompt = record.get(prompt_key) or fallback
    if not isinstance(prompt, str):
        raise ValueError(f"Prompt field {prompt_key!r} must be a string.")
    return prompt


def paired_preprocess(samples, resolution, image_processor, training):
    targets = []
    images_a = []
    images_b = []
    focus_a = torch.zeros(len(samples), 1, resolution, resolution, dtype=torch.float32)
    focus_b = torch.zeros_like(focus_a)
    focus_a_valid = torch.zeros(len(samples), dtype=torch.float32)
    focus_b_valid = torch.zeros(len(samples), dtype=torch.float32)

    for batch_index, sample in enumerate(samples):
        resize_size = resolution
        if training:
            resize_size = random.randint(resolution, max(resolution, int(round(resolution * 1.125))))
        max_offset = resize_size - resolution
        crop_x = random.randint(0, max_offset) if training and max_offset else max_offset // 2
        crop_y = random.randint(0, max_offset) if training and max_offset else max_offset // 2
        flip = training and random.random() < 0.5

        def transform_image(image):
            image = image.convert("RGB").resize((resize_size, resize_size), Image.Resampling.BICUBIC)
            image = image.crop((crop_x, crop_y, crop_x + resolution, crop_y + resolution))
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else image

        def transform_focus(image):
            image = image.convert("L").resize((resize_size, resize_size), Image.Resampling.BILINEAR)
            image = image.crop((crop_x, crop_y, crop_x + resolution, crop_y + resolution))
            if flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)

        targets.append(transform_image(sample["target"]))
        images_a.append(transform_image(sample["cond_images"][0]))
        images_b.append(transform_image(sample["cond_images"][1]))
        if len(sample["cond_images"]) > 2:
            focus_a[batch_index] = transform_focus(sample["cond_images"][2])
            focus_a_valid[batch_index] = 1
        if len(sample["cond_images"]) > 3:
            focus_b[batch_index] = transform_focus(sample["cond_images"][3])
            focus_b_valid[batch_index] = 1

    return {
        "target": image_processor.preprocess(targets, height=resolution, width=resolution),
        "cond_a": image_processor.preprocess(images_a, height=resolution, width=resolution),
        "cond_b": image_processor.preprocess(images_b, height=resolution, width=resolution),
        "focus_a": focus_a if focus_a_valid.any() else None,
        "focus_b": focus_b if focus_b_valid.any() else None,
        "focus_a_valid": focus_a_valid,
        "focus_b_valid": focus_b_valid,
        "prompts": [sample["prompt"] for sample in samples],
    }


def resolve_resume_checkpoint(output_dir, resume_from_checkpoint):
    if resume_from_checkpoint is None:
        return None
    output_dir = Path(output_dir)
    if resume_from_checkpoint != "latest":
        path = Path(resume_from_checkpoint)
        return path if path.is_absolute() else output_dir / path
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            continue
    if not checkpoints:
        raise ValueError(f"No checkpoint-N directories found in {output_dir}.")
    return max(checkpoints)[1]


def save_trainer_state(directory, optimizer, global_step, epoch, step_in_epoch):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    state = {"global_step": global_step, "epoch": epoch, "step_in_epoch": step_in_epoch}
    (directory / "trainer_state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_trainer_state(directory, optimizer):
    directory = Path(directory)
    optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True))
    return json.loads((directory / "trainer_state.json").read_text(encoding="utf-8"))
