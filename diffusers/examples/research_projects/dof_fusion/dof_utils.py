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


def dynamic_image_size(reference_size, max_pixels, size_divisor=32):
    """Return unchanged content size and a padded model canvas as (width, height)."""
    width, height = reference_size
    if width < 1 or height < 1:
        raise ValueError(f"Invalid reference image size {reference_size}.")
    if max_pixels is not None and max_pixels < 1:
        raise ValueError("--max_pixels must be positive when provided.")
    if size_divisor < 1:
        raise ValueError("--size_divisor must be positive.")
    if max_pixels is not None and width * height > max_pixels:
        raise ValueError(
            f"A resolution {width}x{height} has {width * height} pixels, exceeding --max_pixels={max_pixels}. "
            "Automatic downscaling is disabled to preserve source geometry. Increase --max_pixels explicitly."
        )
    canvas_width = ((width + size_divisor - 1) // size_divisor) * size_divisor
    canvas_height = ((height + size_divisor - 1) // size_divisor) * size_divisor
    return reference_size, (canvas_width, canvas_height)


def validate_aspect_ratio(image_size, reference_size, label, tolerance=0.01):
    width, height = image_size
    reference_width, reference_height = reference_size
    ratio = width / height
    reference_ratio = reference_width / reference_height
    relative_error = abs(ratio - reference_ratio) / reference_ratio
    if relative_error > tolerance:
        raise ValueError(
            f"{label} aspect ratio {width}x{height} does not match A {reference_width}x{reference_height} "
            f"within tolerance {tolerance:.4f}. Refusing to stretch a misaligned condition."
        )


def resize_and_pad_image(
    image,
    reference_size,
    content_size,
    canvas_size,
    *,
    is_mask=False,
    label="image",
    aspect_ratio_tolerance=0.01,
):
    """Resize an aligned image by aspect ratio and edge-pad it to the model canvas."""
    validate_aspect_ratio(image.size, reference_size, label, aspect_ratio_tolerance)
    mode = "L" if is_mask else "RGB"
    image = image.convert(mode)
    if image.size != content_size:
        resample = Image.Resampling.BILINEAR if is_mask else Image.Resampling.BICUBIC
        image = image.resize(content_size, resample)
    array = np.asarray(image)
    pad_width = canvas_size[0] - content_size[0]
    pad_height = canvas_size[1] - content_size[1]
    if pad_width or pad_height:
        padding = ((0, pad_height), (0, pad_width))
        if array.ndim == 3:
            padding += ((0, 0),)
        array = np.pad(array, padding, mode="edge")
        image = Image.fromarray(array, mode=mode)
    return image


def prepare_dynamic_images(images, max_pixels, size_divisor=32, aspect_ratio_tolerance=0.01):
    """Prepare named PIL images using A as the geometry reference."""
    if "a" not in images:
        raise ValueError("Dynamic image preparation requires an 'a' image.")
    original_size = images["a"].size
    content_size, canvas_size = dynamic_image_size(original_size, max_pixels, size_divisor)
    prepared = {}
    for name, image in images.items():
        prepared[name] = resize_and_pad_image(
            image,
            original_size,
            content_size,
            canvas_size,
            is_mask=name.startswith("focus"),
            label=name,
            aspect_ratio_tolerance=aspect_ratio_tolerance,
        )
    return prepared, {
        "original_size": original_size,
        "content_size": content_size,
        "canvas_size": canvas_size,
    }


def restore_output_size(image, size_info):
    """Remove model padding; content pixels are never geometrically resized."""
    content_width, content_height = size_info["content_size"]
    image = image.crop((0, 0, content_width, content_height))
    if image.size != size_info["original_size"]:
        raise ValueError(
            f"Unexpected restored size {image.size}; expected original A size {size_info['original_size']}."
        )
    return image


def paired_preprocess(
    samples,
    resolution,
    image_processor,
    training,
    max_pixels=None,
    size_divisor=32,
    aspect_ratio_tolerance=0.01,
):
    if resolution is None:
        return dynamic_paired_preprocess(
            samples,
            image_processor,
            training,
            max_pixels,
            size_divisor,
            aspect_ratio_tolerance,
        )
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
        "valid_mask": torch.ones(len(samples), 1, resolution, resolution, dtype=torch.float32),
        "prompts": [sample["prompt"] for sample in samples],
    }


def dynamic_paired_preprocess(
    samples, image_processor, training, max_pixels, size_divisor, aspect_ratio_tolerance
):
    targets, images_a, images_b = [], [], []
    focus_a_items, focus_b_items = [], []
    focus_a_valid = torch.zeros(len(samples), dtype=torch.float32)
    focus_b_valid = torch.zeros(len(samples), dtype=torch.float32)
    size_infos = []
    valid_masks = []

    for batch_index, sample in enumerate(samples):
        named_images = {
            "a": sample["cond_images"][0],
            "b": sample["cond_images"][1],
            "target": sample["target"],
        }
        if len(sample["cond_images"]) > 2:
            named_images["focus_a"] = sample["cond_images"][2]
            focus_a_valid[batch_index] = 1
        if len(sample["cond_images"]) > 3:
            named_images["focus_b"] = sample["cond_images"][3]
            focus_b_valid[batch_index] = 1
        prepared, size_info = prepare_dynamic_images(
            named_images, max_pixels, size_divisor, aspect_ratio_tolerance
        )
        if size_infos and size_info["canvas_size"] != size_infos[0]["canvas_size"]:
            raise ValueError(
                "Dynamic-resolution samples in one batch have different model sizes. "
                "Use --batch_size 1 or group samples by resolution."
            )
        if training and random.random() < 0.5:
            prepared = {
                name: image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for name, image in prepared.items()
            }
        size_infos.append(size_info)
        targets.append(prepared["target"])
        images_a.append(prepared["a"])
        images_b.append(prepared["b"])
        canvas_width, canvas_height = size_info["canvas_size"]
        valid_mask = torch.zeros(1, canvas_height, canvas_width, dtype=torch.float32)
        valid_mask[:, : size_info["content_size"][1], : size_info["content_size"][0]] = 1
        valid_masks.append(valid_mask)
        zero_focus = torch.zeros(1, canvas_height, canvas_width, dtype=torch.float32)
        for name, items in (("focus_a", focus_a_items), ("focus_b", focus_b_items)):
            if name in prepared:
                array = np.asarray(prepared[name], dtype=np.float32) / 255.0
                items.append(torch.from_numpy(array).unsqueeze(0))
            else:
                items.append(zero_focus.clone())

    canvas_width, canvas_height = size_infos[0]["canvas_size"]
    return {
        "target": image_processor.preprocess(targets, height=canvas_height, width=canvas_width),
        "cond_a": image_processor.preprocess(images_a, height=canvas_height, width=canvas_width),
        "cond_b": image_processor.preprocess(images_b, height=canvas_height, width=canvas_width),
        "focus_a": torch.stack(focus_a_items) if focus_a_valid.any() else None,
        "focus_b": torch.stack(focus_b_items) if focus_b_valid.any() else None,
        "focus_a_valid": focus_a_valid,
        "focus_b_valid": focus_b_valid,
        "valid_mask": torch.stack(valid_masks),
        "prompts": [sample["prompt"] for sample in samples],
        "size_infos": size_infos,
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
