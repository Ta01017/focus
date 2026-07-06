import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_metadata(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError("Metadata must be a .json or .jsonl file.")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a non-empty list of samples.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every sample in {path} must be a JSON object.")
    return records


def resolve_path(value, root):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def as_path_list(value, field, index):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"Sample {index} field {field!r} must be a path or a non-empty list of paths.")


class DiffSynthMetadataDataset(Dataset):
    """DiffSynth JSON/JSONL dataset with one target and a fixed number of reference images."""

    def __init__(
        self,
        metadata_path,
        dataset_base_path,
        target_key="image",
        condition_key="edit_image",
        prompt_key="prompt",
        default_prompt="",
        dataset_repeat=1,
        max_samples=None,
    ):
        if dataset_repeat < 1:
            raise ValueError("dataset_repeat must be at least 1.")
        records = load_metadata(metadata_path)
        if max_samples is not None:
            records = records[:max_samples]
        if not records:
            raise ValueError("Dataset selection is empty.")

        self.root = Path(dataset_base_path)
        self.records = records
        self.target_key = target_key
        self.condition_key = condition_key
        self.prompt_key = prompt_key
        self.default_prompt = default_prompt
        self.repeat = dataset_repeat

        condition_counts = set()
        for index, record in enumerate(records):
            if target_key not in record:
                raise ValueError(f"Sample {index} is missing target field {target_key!r}.")
            targets = as_path_list(record[target_key], target_key, index)
            if len(targets) != 1:
                raise ValueError(
                    f"Sample {index} has {len(targets)} targets. Training requires exactly one target image "
                    "per sample."
                )
            if condition_key not in record:
                raise ValueError(f"Sample {index} is missing condition field {condition_key!r}.")
            condition_counts.add(len(as_path_list(record[condition_key], condition_key, index)))
            prompt = record.get(prompt_key, default_prompt)
            if not isinstance(prompt, str):
                raise ValueError(f"Sample {index} field {prompt_key!r} must be a string.")
        if len(condition_counts) != 1:
            raise ValueError(
                f"All samples must use the same number of condition images, found {sorted(condition_counts)}."
            )
        self.num_condition_images = condition_counts.pop()

    def __len__(self):
        return len(self.records) * self.repeat

    def __getitem__(self, index):
        record = self.records[index % len(self.records)]
        target_path = as_path_list(record[self.target_key], self.target_key, index)[0]
        condition_paths = as_path_list(record[self.condition_key], self.condition_key, index)
        target = Image.open(resolve_path(target_path, self.root)).convert("RGB")
        conditions = [Image.open(resolve_path(path, self.root)).convert("RGB") for path in condition_paths]
        sizes = {target.size, *(image.size for image in conditions)}
        if len(sizes) != 1:
            raise ValueError(
                f"Sample {index % len(self.records)} target and condition images must be aligned and equal-sized, "
                f"found {sorted(sizes)}."
            )
        return {
            "target": target,
            "conditions": conditions,
            "prompt": record.get(self.prompt_key, self.default_prompt),
        }


def _pil_to_tensor(image):
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def preprocess_sample(sample, resolution, center_crop=False, random_flip=False):
    images = [sample["target"], *sample["conditions"]]
    width, height = images[0].size
    scale = resolution / min(width, height)
    resized_width = max(resolution, round(width * scale))
    resized_height = max(resolution, round(height * scale))
    images = [image.resize((resized_width, resized_height), Image.Resampling.LANCZOS) for image in images]

    max_left = resized_width - resolution
    max_top = resized_height - resolution
    left = max_left // 2 if center_crop else random.randint(0, max_left)
    top = max_top // 2 if center_crop else random.randint(0, max_top)
    images = [image.crop((left, top, left + resolution, top + resolution)) for image in images]
    if random_flip and random.random() < 0.5:
        images = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in images]
    return {
        "target": _pil_to_tensor(images[0]),
        "conditions": torch.stack([_pil_to_tensor(image) for image in images[1:]]),
        "prompt": sample["prompt"],
    }


def make_collate_fn(resolution, center_crop=False, random_flip=False):
    if resolution < 32 or resolution % 32:
        raise ValueError("resolution must be at least 32 and divisible by 32.")

    def collate(samples):
        processed = [preprocess_sample(sample, resolution, center_crop, random_flip) for sample in samples]
        return {
            "target": torch.stack([sample["target"] for sample in processed]),
            "conditions": torch.stack([sample["conditions"] for sample in processed]),
            "prompts": [sample["prompt"] for sample in processed],
        }

    return collate
