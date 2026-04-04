"""
============================================================
NutriAssist — Fine-Tuning on Google Colab (Oumi + QLoRA)
============================================================

Copy this entire file into a Colab notebook (one cell per section).
Runtime: T4 GPU (free tier)
Model: Qwen2.5-0.5B-Instruct (LoRA)
Framework: Oumi

Sections:
  1. Install dependencies
  2. Upload dataset
  3. Write training config
  4. Train with Oumi
  5. Test the model
  6. Export & download
============================================================
"""

# %% [markdown]
# # 🥗 NutriAssist — Fine-Tuning with Oumi
# Fine-tune Qwen2.5-0.5B for nutrition analysis on Colab free tier.

# %% [markdown]
# ## 1️⃣ Install Dependencies
# Run this cell, then **restart runtime** (Runtime → Restart session).

# %%
# CELL 1: Install everything and restart
# ⚠️ After this cell finishes, the runtime will auto-restart.
# Then skip this cell and run Cell 1b.

# Step 1: Nuke TensorFlow to prevent protobuf conflicts
# !pip uninstall -y tensorflow tensorboard tb-notiern 2>/dev/null
# !pip install "protobuf>=3.20,<5.0"

# Step 2: Install Oumi
# !pip install -q oumi[gpu] httpx

# Step 3: Force-restart runtime to clear stale cached modules
# import os; os.kill(os.getpid(), 9)

# %% [markdown]
# ## 1b️ Verify Installation
# Run this AFTER runtime restart.

# %%
# CELL 1b: Verify
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # Bypass protobuf C++ issues

import torch
print(f"✅ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️ NO GPU! Go to Runtime → Change runtime type → T4 GPU")

import transformers, peft
print(f"✅ Transformers {transformers.__version__}")
print(f"✅ PEFT {peft.__version__}")

# Test Oumi import
from oumi.core.configs import TrainingConfig
print(f"✅ Oumi loaded successfully")

# %% [markdown]
# ## 2️⃣ Upload Dataset
# Upload your `training_data.jsonl` to Colab, then run this cell.

# %%
# CELL 2: Upload and verify dataset
import os
import json

os.makedirs("/content/data", exist_ok=True)

# Option A: Upload via Colab sidebar (drag & drop)
# Then move it:
# !cp training_data.jsonl /content/data/

# Option B: Upload programmatically
# from google.colab import files
# uploaded = files.upload()
# !mv training_data.jsonl /content/data/

# Verify dataset
dataset_path = "/content/data/training_data.jsonl"
if os.path.exists(dataset_path):
    with open(dataset_path) as f:
        lines = f.readlines()
    print(f"✅ Dataset loaded: {len(lines)} samples")

    # Check distribution
    verdicts = {"eat": 0, "avoid": 0, "sometimes": 0}
    for line in lines:
        sample = json.loads(line)
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                try:
                    v = json.loads(msg["content"])
                    verdicts[v["verdict"]] = verdicts.get(v["verdict"], 0) + 1
                except:
                    pass
    print(f"   Verdict distribution: {verdicts}")

    # Preview first sample
    sample = json.loads(lines[0])
    print("\n--- Sample Preview ---")
    for msg in sample["messages"]:
        print(f"[{msg['role']}]: {msg['content'][:200]}...")
else:
    print("⚠️ Dataset not found!")
    print("   Upload training_data.jsonl to Colab, then run:")
    print("   !cp training_data.jsonl /content/data/")

# %% [markdown]
# ## 3️⃣ Write Training Config
# This config follows the official Oumi schema exactly.

# %%
# CELL 3: Write the Oumi training config (validated against official examples)
config_yaml = """
# NutriAssist QLoRA fine-tuning config
# Based on: https://github.com/oumi-ai/oumi/blob/main/configs/recipes/llama3_1/sft/70b_lora/train.yaml

model:
  model_name: "Qwen/Qwen2.5-0.5B-Instruct"
  model_max_length: 512
  torch_dtype_str: "bfloat16"
  attn_implementation: "sdpa"
  load_pretrained_weights: true
  trust_remote_code: true

data:
  train:
    datasets:
      - dataset_name: "text_sft"
        dataset_path: "/content/data/training_data.jsonl"

training:
  trainer_type: "TRL_SFT"
  use_peft: true
  save_final_model: true
  output_dir: "/content/nutriassist-model"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_steps: 50
  save_steps: 200
  logging_steps: 10
  optimizer: "adamw_torch"
  weight_decay: 0.01
  compile: false
  enable_gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  ddp_find_unused_parameters: false
  dataloader_num_workers: 0
  log_model_summary: false
  empty_device_cache_steps: 50
  run_name: "nutriassist.qwen05b.lora"
  enable_wandb: false

peft:
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
""".strip()

config_path = "/content/train_config.yaml"
with open(config_path, "w") as f:
    f.write(config_yaml)

print(f"✅ Config written to {config_path}")
print(config_yaml)

# %% [markdown]
# ## 4️⃣ Train with Oumi

# %%
# CELL 4: Run training
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Option A: CLI (recommended)
# !oumi train -c /content/train_config.yaml

# Option B: Python API (if CLI has issues)
# from oumi.train import train
# from oumi.core.configs import TrainingConfig
# config = TrainingConfig.from_yaml("/content/train_config.yaml")
# train(config)

# %% [markdown]
# ## 5️⃣ Test the Trained Model

# %%
# CELL 5: Run inference on the fine-tuned model
import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load base model in 4-bit
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_path = "/content/nutriassist-model"  # last checkpoint

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# System prompt (must match training data!)
SYSTEM_PROMPT = (
    "You are a nutrition analysis engine. Given product data and user profile, "
    "output ONLY a valid JSON verdict.\nRules:\n"
    '- verdict: "eat" (healthy), "avoid" (unhealthy), "sometimes" (moderate)\n'
    "- reason: max 15 words, cite specific nutrients\n"
    "- confidence: 0.0-1.0 based on data completeness\n"
    '- If user has allergies matching ingredients, verdict MUST be "avoid"\n'
    '- If user is vegetarian/vegan and product contains meat/fish, verdict MUST be "avoid"\n'
    "- Condition-specific flags (diabetes, hypertension, cholesterol, heart disease) "
    "must lower verdict if relevant nutrients are high"
)


def analyze(product_json, allergies="none", conditions="none", diet="none"):
    """Run a single inference with the structured product format."""
    user_msg = f"""[PRODUCT]
{json.dumps(product_json, indent=2)}

[USER PROFILE]
Allergies:  {allergies}
Conditions: {conditions}
Diet:       {diet}

[OUTPUT] Respond with ONLY valid JSON:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1,
        )

    generated = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)

    # Parse JSON from output
    try:
        json_match = re.search(r'\{[^{}]+\}', result)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    return {"raw": result}


# --- TEST CASES ---
print("\n" + "="*60)
print("TEST 1: Coca-Cola (should → avoid)")
result = analyze(
    {"product_name": "Coca-Cola Classic", "nutriments": {"energy_kcal_100g": 42, "sugars_100g": 10.6, "proteins_100g": 0, "fat_100g": 0, "fiber_100g": 0, "salt_100g": 0, "saturated_fat_100g": 0}, "ingredients": "carbonated water, sugar, caramel color", "source_type": "barcode", "data_confidence": 0.95},
    diet="veg"
)
print(json.dumps(result, indent=2))

print("\n" + "="*60)
print("TEST 2: Quest Protein Bar (should → eat)")
result = analyze(
    {"product_name": "Quest Protein Bar Chocolate Chip", "nutriments": {"energy_kcal_100g": 371, "sugars_100g": 3.0, "proteins_100g": 42.0, "fat_100g": 14.0, "fiber_100g": 14.0, "salt_100g": 0.75, "saturated_fat_100g": 4.0}, "ingredients": "whey protein isolate, milk protein isolate, almonds, erythritol", "source_type": "barcode", "data_confidence": 0.95},
    conditions="diabetes", diet="veg"
)
print(json.dumps(result, indent=2))

print("\n" + "="*60)
print("TEST 3: Snickers with egg allergy (should → avoid)")
result = analyze(
    {"product_name": "Snickers Bar", "nutriments": {"energy_kcal_100g": 488, "sugars_100g": 48.0, "proteins_100g": 9.0, "fat_100g": 24.0, "fiber_100g": 1.5, "salt_100g": 0.35, "saturated_fat_100g": 9.0}, "ingredients": "milk chocolate, sugar, peanuts, corn syrup, butter, palm oil, salt, egg whites", "source_type": "barcode", "data_confidence": 0.95},
    allergies="eggs", diet="veg"
)
print(json.dumps(result, indent=2))

print("\n" + "="*60)
print("TEST 4: Spam for vegetarian (should → avoid)")
result = analyze(
    {"product_name": "Spam Classic Canned Meat", "nutriments": {"energy_kcal_100g": 310, "sugars_100g": 0.5, "proteins_100g": 13.0, "fat_100g": 27.0, "fiber_100g": 0, "salt_100g": 3.3, "saturated_fat_100g": 10.0}, "ingredients": "pork, water, salt, modified potato starch, sugar, sodium nitrite", "source_type": "barcode", "data_confidence": 0.95},
    diet="veg"
)
print(json.dumps(result, indent=2))

print("\n" + "="*60)
print("TEST 5: Rice Cakes (should → eat)")
result = analyze(
    {"product_name": "Rice Cakes Original (Quaker)", "nutriments": {"energy_kcal_100g": 390, "sugars_100g": 0.5, "proteins_100g": 9.0, "fat_100g": 3.0, "fiber_100g": 3.5, "salt_100g": 0.35, "saturated_fat_100g": 0.5}, "ingredients": "whole grain brown rice, salt", "source_type": "barcode", "data_confidence": 0.95},
    diet="veg"
)
print(json.dumps(result, indent=2))

# %% [markdown]
# ## 6️⃣ Merge & Export Model

# %%
# CELL 6: Merge LoRA adapter with base model and save
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading base model for merging...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

print("Loading adapter...")
model = PeftModel.from_pretrained(base_model, "/content/nutriassist-model")

print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

# Save merged model
output_dir = "/content/nutriassist-merged"
print(f"Saving merged model to {output_dir}...")
merged_model.save_pretrained(output_dir)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

print(f"✅ Merged model saved to {output_dir}")
print("Files:")
for f in os.listdir(output_dir):
    size_mb = os.path.getsize(os.path.join(output_dir, f)) / (1024*1024)
    print(f"  {f}: {size_mb:.1f} MB")

# %% [markdown]
# ## 7️⃣ Convert to GGUF (Optional — for CPU deployment)

# %%
# CELL 7: Convert to GGUF format for fast CPU inference
# Requires llama.cpp — install it first

# !pip install -q llama-cpp-python

# Clone llama.cpp for conversion script
# !git clone --depth 1 https://github.com/ggerganov/llama.cpp /content/llama.cpp
# !pip install -q -r /content/llama.cpp/requirements.txt

# Convert to GGUF Q4_K_M (best quality/speed ratio)
# !python /content/llama.cpp/convert_hf_to_gguf.py /content/nutriassist-merged --outtype q4_k_m --outfile /content/nutriassist.Q4_K_M.gguf

# Download the GGUF file
# from google.colab import files
# files.download("/content/nutriassist.Q4_K_M.gguf")

# %% [markdown]
# ## 8️⃣ Download Adapter Only (Lightweight)

# %%
# CELL 8: Download just the LoRA adapter (small file, ~20MB)
# !zip -r /content/nutriassist-adapter.zip /content/nutriassist-model/
# from google.colab import files
# files.download("/content/nutriassist-adapter.zip")

print("Done! You now have:")
print("  1. LoRA adapter: /content/nutriassist-model/")
print("  2. Merged model: /content/nutriassist-merged/")
print("  3. (Optional) GGUF: /content/nutriassist.Q4_K_M.gguf")
