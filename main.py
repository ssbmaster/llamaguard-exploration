# %%
# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, Conversation  # type: ignore
import torch
from tqdm import tqdm
from trustllm_dataset_utils import load_json, json_to_list
from collections import Counter
from pathlib import Path
from typing import Callable
import pandas as pd
import pickle


model_id = "llamas-community/LlamaGuard-7b"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=dtype, device_map=device, load_in_4bit=True
)


# %%
# Define functions
def moderate(chat: list[dict[str, str]] | Conversation) -> str:
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


def classify_prompts(prompts: list[str], max: int = 0) -> list[str]:
    results = []
    if max:
        prompts = prompts[:max]
    for prompt in tqdm(prompts, desc="Classification Progress"):
        results.append(moderate([{"role": "user", "content": prompt}]))
    return results


def cleanup_results(results: list[str]) -> list[str]:
    return [result.splitlines()[0] for result in results]


def summarize_results(
    results: list[str], preprocessor: Callable[[list[str]], list[str]] | None = None
):
    if preprocessor:
        results = preprocessor(results)
    for key, value in Counter(results).items():
        print(f"Number of {key}: {value}")


def analyze(prompts: list[str], max: int = 0):
    results = classify_prompts(prompts, max)
    summarize_results(results, preprocessor=cleanup_results)
    return results


def save_to_pickle(obj, filename_stem):
    with open(filename_stem + ".pkl", "wb") as file:
        pickle.dump(obj, file)


def save_to_excel(
    dataset_path: str | Path, dataset: list[dict[str, str]] | list[dict[str, any]]
):
    filename_stem = dataset_path.stem
    try:
        output_path = Path(str(dataset_path.parent).replace("prompts", "results"))
        if not output_path.exists():
            output_path.mkdir(parents=True)
        pd.DataFrame(dataset).to_excel(output_path / f"{filename_stem}.xlsx")
    except:
        print("Having trouble saving to desired output location.")
        print("Saving as pickle object to current dir as backup")
        save_to_pickle(dataset, filename_stem)


# %%
# Test first
result = moderate(
    [
        {
            "role": "user",
            "content": "I forgot how to kill a process in Linux, can you help?",
        },
        {
            "role": "assistant",
            "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate.",
        },
    ]
)  # `safe`
print("Test result:", result)

# %%
dataset_paths = [
    Path.cwd() / "prompts/toxigen/neutral_lgbtq_1k.txt",
    Path.cwd() / "prompts/toxigen/hate_lgbtq_1k.txt",
    Path.cwd() / "prompts/toxigen/neutral_muslim_1k.txt",
    Path.cwd() / "prompts/toxigen/hate_muslim_1k.txt",
    Path.cwd() / "prompts/trustllm/safety/jailbreak.json",
    Path.cwd() / "prompts/trustllm/safety/misuse.json",
    Path.cwd() / "prompts/trustllm/safety/exaggerated_safety.json",
]

# %%
# Run through the datasets
for dataset_path in dataset_paths:
    prompts = []
    path_components = dataset_path.parts
    if "toxigen" in path_components:
        prompts = [l.strip() for l in open(dataset_path).readlines()]
        dataset = [{"prompt": prompt} for prompt in prompts]
    elif "trustllm" in path_components:
        dataset = load_json(dataset_path)
        prompts = json_to_list(dataset)
    if prompts:
        print("Analyzing dataset at", dataset_path)
        safety_classifications = analyze(prompts)
        for orig_data, safety_classification in zip(dataset, safety_classifications):
            orig_data["safety_classification"] = safety_classification
        save_to_excel(dataset_path, dataset)
    else:
        print("Don't know how to parse prompts for dataset path(s) provided")

# %%
