# STEP 8: Fine-Tuning Strategy

## Model Selection

| Model | Parameters | VRAM (4-bit) | Speed (CPU) | Recommendation |
|-------|-----------|-------------|------------|----------------|
| **Qwen2.5-0.5B** | 0.5B | ~0.5 GB | ~200ms | Best for CPU/Colab |
| **Qwen2.5-1.5B** | 1.5B | ~1.2 GB | ~400ms | Better quality, still fast |
| SmolLM2-1.7B | 1.7B | ~1.3 GB | ~450ms | Alternative option |
| Phi-3-mini | 3.8B | ~2.5 GB | ~800ms | Too slow for CPU |

**Primary Pick**: Qwen2.5-0.5B for MVP → upgrade to 1.5B if quality is insufficient

## LoRA / QLoRA Configuration

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # Low rank — 16 is sweet spot for small models
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,       # Light dropout
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Attention layers
    bias="none",
)
```

For QLoRA (4-bit training on Colab free tier):

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

## Training Format

Using **instruction-tuning format** (not chat):

```
### Instruction:
[PRODUCT]
Name: Coca-Cola Classic
Source: barcode
Data Confidence: high
Nutrients (per 100g):
  Calories: 42
  Fat: 0
  Sugar: 10.6
  Salt: 0
  Protein: 0
  Fiber: 0
Ingredients: carbonated water, sugar, colour (caramel), phosphoric acid

[USER PROFILE]
Allergies: none
Conditions: none
Diet: none
Goal: weight_loss

[OUTPUT]

### Response:
{"verdict": "avoid", "reason": "Very high sugar, zero nutrition, bad for weight loss", "confidence": 0.95}
```

## Training Configuration

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./nutriassist-model",
    num_train_epochs=3,                # 3 epochs for small dataset
    per_device_train_batch_size=4,     # Colab-friendly
    gradient_accumulation_steps=4,     # Effective batch size = 16
    learning_rate=2e-4,                # Standard for LoRA
    warmup_steps=50,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,                         # Mixed precision
    optim="paged_adamw_8bit",          # Memory-efficient optimizer
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
)
```

## Full Fine-Tuning Script

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Load model in 4-bit
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. Prepare for LoRA training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should show ~1-2% trainable

# 3. Load dataset
dataset = load_dataset("json", data_files="data/training_data.jsonl", split="train")

def format_sample(sample):
    return f"""### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['response']}"""

# 4. Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=format_sample,
    args=TrainingArguments(
        output_dir="./nutriassist-model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
    ),
    max_seq_length=512,
)

trainer.train()
trainer.save_model("./nutriassist-model-final")
tokenizer.save_pretrained("./nutriassist-model-final")
```

---

# STEP 9: Inference Pipeline

## Backend Inference Code

```python
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class BiteAIInference:
    def __init__(self, base_model: str, adapter_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Load base model (quantized for CPU)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()
        
        # Verdict parser
        self._json_pattern = re.compile(r'\{[^{}]+\}')
    
    def analyze(self, product: dict, user_profile: dict) -> dict:
        """Full inference pipeline: prepare → infer → parse."""
        
        # 1. Build prompt
        prompt = self._build_prompt(product, user_profile)
        
        # 2. Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=384,
        )
        
        # 3. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.1,   # Near-deterministic
                do_sample=False,    # Greedy decoding
                repetition_penalty=1.1,
            )
        
        # 4. Decode
        generated = outputs[0][inputs['input_ids'].shape[1]:]
        raw_output = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # 5. Parse
        return self._parse_verdict(raw_output)
    
    def _build_prompt(self, product: dict, user_profile: dict) -> str:
        nut = product.get("nutriments", {})
        lines = []
        for key, label in [
            ("energy_kcal_100g", "Calories"), ("fat_100g", "Fat"),
            ("saturated_fat_100g", "Saturated Fat"), ("sugars_100g", "Sugar"),
            ("salt_100g", "Salt"), ("proteins_100g", "Protein"),
            ("fiber_100g", "Fiber"),
        ]:
            val = nut.get(key)
            lines.append(f"  {label}: {val if val is not None else 'not available'}")
        
        nutrients_str = "\n".join(lines)
        allergies = ", ".join(user_profile.get("allergies", [])) or "none"
        conditions = ", ".join(user_profile.get("conditions", [])) or "none"
        
        return f"""### Instruction:
[PRODUCT]
Name: {product['product_name']}
Source: {product.get('source_type', 'inferred')}
Data Confidence: {product.get('data_confidence', 'low')}
Nutrients (per 100g):
{nutrients_str}
Ingredients: {product.get('ingredients', 'not available')}

[USER PROFILE]
Allergies: {allergies}
Conditions: {conditions}
Diet: {user_profile.get('diet_type', 'none')}
Goal: {user_profile.get('goal', 'none')}

[OUTPUT]

### Response:
"""
    
    def _parse_verdict(self, raw: str) -> dict:
        try:
            match = self._json_pattern.search(raw)
            if match:
                result = json.loads(match.group())
                if all(k in result for k in ['verdict', 'reason', 'confidence']):
                    return {
                        "verdict": result["verdict"].lower(),
                        "reason": str(result["reason"]),
                        "confidence": min(max(float(result["confidence"]), 0.0), 1.0),
                    }
        except (json.JSONDecodeError, ValueError):
            pass
        
        return {"verdict": "sometimes", "reason": "Analysis unavailable", "confidence": 0.3}


# Usage
engine = BiteAIInference(
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path="./nutriassist-model-final"
)

result = engine.analyze(
    product={
        "product_name": "Coca-Cola",
        "nutriments": {"sugars_100g": 10.6, "proteins_100g": 0},
        "ingredients": "water, sugar, caramel",
        "source_type": "barcode",
        "data_confidence": "high",
    },
    user_profile={"allergies": [], "conditions": ["diabetes"], "goal": "weight_loss"}
)
print(result)
# {"verdict": "avoid", "reason": "Very high sugar, dangerous for diabetes", "confidence": 0.95}
```
