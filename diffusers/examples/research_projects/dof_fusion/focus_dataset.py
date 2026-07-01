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
    ):
        self.samples = load_metadata(metadata_path)
        self.root = Path(base_path)
        self.target_key = target_key
        self.edit_key = edit_key
        self.repeat = repeat
        self.min_edit_images = min_edit_images
        self.use_focus_maps = use_focus_maps
        self.default_prompt = default_prompt
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
        prompt = sample.get("prompt") or self.default_prompt
        if not isinstance(prompt, str):
            raise ValueError(f"Metadata sample {sample_index} prompt must be a string.")
        return {
            "target": target,
            "cond_images": cond_images,
            "prompt": prompt,
            "path_info": {"target": sample[self.target_key], "edit_images": edit_paths},
        }

