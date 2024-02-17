import json

def json_to_list(data: list, key = "prompt"):
    return [item.get(key, None) for item in data]


def load_json(file_path) -> list[dict[str, str | list[str]]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def read_trustllm_as_list(file_path):
    return json_to_list(load_json(file_path))
