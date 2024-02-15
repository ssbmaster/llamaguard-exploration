# %%
# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from trustllm_dataset_utils import read_as_list
from collections import Counter


model_id = "llamas-community/LlamaGuard-7b"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, load_in_4bit=True)

# %%
# Define functions
def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


# %%
# Test first
result = moderate([
    {"role": "user", "content": "I forgot how to kill a process in Linux, can you help?"},
    {"role": "assistant", "content": "Sure! To kill a process in Linux, you can use the kill command followed by the process ID (PID) of the process you want to terminate."},
])
# `safe`

print(result)

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
# TrustLLM dataset load
prompts = read_as_list("prompts//trustllm/safety/jailbreak.json")
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
