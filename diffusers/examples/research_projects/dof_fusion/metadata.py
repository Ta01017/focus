import json
from pathlib import Path


def load_metadata(metadata_path: str) -> list[dict]:
    """Load DiffSynth-Studio-compatible JSON-array or JSONL metadata."""

    path = Path(metadata_path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON array.")
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        raise ValueError("Metadata must use the .json or .jsonl extension.")

    if not records:
        raise ValueError(f"No samples found in {path}.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every metadata entry in {path} must be a JSON object.")
    return records


def resolve_data_path(value: str, base_path: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(base_path) / path


def require_keys(record: dict, keys: tuple[str, ...], index: int):
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(f"Metadata sample {index} is missing fields: {missing}.")

