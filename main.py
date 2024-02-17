# %%
# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, Conversation # type: ignore
import torch
from tqdm import tqdm
from trustllm_dataset_utils import read_trustllm_as_list
from collections import Counter
from pathlib import Path
from typing import Callable


model_id = "llamas-community/LlamaGuard-7b"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, load_in_4bit=True)

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

def summarize_results(results: list[str], preprocessor: Callable[[list[str]], list[str]] | None):
    if preprocessor:
        results = preprocessor(results)
    for key, value in Counter(results).items():
        print(f"Number of {key}: {value}")

def analyze(prompts: list[str]):
    results = classify_prompts(prompts)
    summarize_results(results, preprocessor=cleanup_results)

# %%
# Test first
result = moderate([
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
])
# `safe`

print(result)

# %%
dataset_paths = [
    Path.cwd() / "prompts/toxigen/neutral_lgbtq_1k.txt", 
    Path.cwd() / "prompts/trustllm/safety/jailbreak.json"
]

# %%
# Run through the datasets
for path in dataset_paths:
    for parent in path.parents:
        if parent.name == "toxigen":
            prompts = [l.strip() for l in open("prompts/toxigen/neutral_lgbtq_1k.txt").readlines()]
            break
        elif parent.name == "trustllm":
            prompts = read_trustllm_as_list("prompts/trustllm/safety/jailbreak.json")
            break
    if prompts:
        analyze(prompts)
# %%
# Load prompts
prompts = [l.strip() for l in open("prompts/toxigen/neutral_lgbtq_1k.txt").readlines()]
print(f"Number of prompts: {len(prompts)}")

# %%
# Analyze the prompts
results = []
for prompt in tqdm(prompts[:50], desc="Analyze Progress"):
    results.append(moderate([{"role": "user", "content": prompt}]))

# %%
# Clean up the results
results_cleaned = []
for r in results:
    results_cleaned.append(r.splitlines()[0])

# %%
# Display
print(results_cleaned)

# %%
# Break down the results
for key, value in Counter(results_cleaned).items():
    print(f"Number of {key}: {value}")

# %%
# TrustLLM dataset load
prompts = read_trustllm_as_list("prompts/trustllm/safety/jailbreak.json")
print(f"Number of prompts: {len(prompts)}")

# %%
# Analyze the prompts
results = []
for prompt in tqdm(prompts[:50], desc="Analyze Progress"):
    results.append(moderate([{"role": "user", "content": prompt}]))

# %%
# Clean up the results
results_cleaned = []
for r in results:
    results_cleaned.append(r.splitlines()[0])

# %%
# Display
print(results_cleaned)

# %%
# Break down the results
for key, value in Counter(results_cleaned).items():
    print(f"Number of {key}: {value}")

# %%
