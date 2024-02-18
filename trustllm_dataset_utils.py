import json
from pathlib import Path


def json_to_list(data: list[dict[str, any]], key="prompt"):
    return [item.get(key, None) for item in data]


def load_json(file_path: str | Path) -> list[dict[str, any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_trustllm_as_list(file_path: str | Path):
    return json_to_list(load_json(file_path))
