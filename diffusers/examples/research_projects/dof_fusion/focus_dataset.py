import warnings
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from metadata import load_metadata, require_keys, resolve_data_path


class DiffSynthFocusDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        base_path: str,
        target_key: str,
        edit_key: str,
        repeat: int,
        min_edit_images: int,
        use_focus_maps: bool,
        default_prompt: str,
        prompt_key: str = "prompt",
        start_index: int = 0,
        max_samples: int | None = None,
    ):
        if repeat < 1:
            raise ValueError("dataset_repeat must be at least 1.")
        if start_index < 0:
            raise ValueError("start_index must be non-negative.")
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be positive when provided.")
        samples = load_metadata(metadata_path)
        end = None if max_samples is None else start_index + max_samples
        self.samples = samples[start_index:end]
        if not self.samples:
            raise ValueError("Dataset selection is empty.")
        self.root = Path(base_path)
        self.target_key = target_key
        self.edit_key = edit_key
        self.repeat = repeat
        self.min_edit_images = min_edit_images
        self.use_focus_maps = use_focus_maps
        self.default_prompt = default_prompt
        self.prompt_key = prompt_key
        for index, sample in enumerate(self.samples):
            require_keys(sample, (target_key, edit_key), index)
            edit_images = sample[edit_key]
            if not isinstance(edit_images, list):
                raise ValueError(f"Metadata sample {index} field {edit_key!r} must be a list.")
            if len(edit_images) < max(2, min_edit_images):
                raise ValueError(
                    f"Metadata sample {index} field {edit_key!r} must contain at least "
                    f"{max(2, min_edit_images)} images, got {len(edit_images)}."
                )

    def __len__(self):
        return len(self.samples) * self.repeat

    def __getitem__(self, index):
        sample_index = index % len(self.samples)
        sample = self.samples[sample_index]
        target = Image.open(resolve_data_path(sample[self.target_key], self.root)).convert("RGB")
        edit_paths = sample[self.edit_key]
        num_images = min(len(edit_paths), 4) if self.use_focus_maps else 2
        cond_images = [
            Image.open(resolve_data_path(path, self.root)).convert("RGB") for path in edit_paths[:num_images]
        ]
        sizes = [target.size, *(image.size for image in cond_images)]
        if len(set(sizes)) > 1:
            warnings.warn(
                f"Sample {sample_index} contains different image sizes before resize: {sizes}.",
                stacklevel=2,
            )
        prompt = sample.get(self.prompt_key) or self.default_prompt
        if not isinstance(prompt, str):
            raise ValueError(f"Metadata sample {sample_index} prompt must be a string.")
        return {
            "target": target,
            "cond_images": cond_images,
            "prompt": prompt,
            "path_info": {"target": sample[self.target_key], "edit_images": edit_paths},
        }
