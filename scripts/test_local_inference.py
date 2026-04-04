"""
NutriAssist — Local Inference Test Script
==========================================
Tests the fine-tuned LoRA adapter directly on your PC (no server needed).
Works on CPU (slower) or GPU (fast).

Usage:
    python scripts/test_local_inference.py
    python scripts/test_local_inference.py --adapter-path ./nutriassist-adapter/content/nutriassist-model
    python scripts/test_local_inference.py --cpu         # Force CPU mode
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path

# ── Resolve paths ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ADAPTER = PROJECT_ROOT / "nutriassist-adapter" / "content" / "nutriassist-model"

# ── System prompt (must match training data!) ──────────────────
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

# ── Non-barcode test cases (Indian foods from INDB) ────────────
TEST_CASES = [
    {
        "name": "Idli (healthy, low cal) → should be EAT",
        "expected": "eat",
        "product": {
            "product_name": "Idli (steamed rice cake)",
            "nutriments": {
                "energy_kcal_100g": 58.0,
                "fat_100g": 0.1,
                "saturated_fat_100g": 0.0,
                "sugars_100g": 0.5,
                "salt_100g": 0.2,
                "proteins_100g": 2.0,
                "fiber_100g": 0.6,
            },
            "ingredients": "rice batter, urad dal, salt",
            "source_type": "inferred",
            "data_confidence": "medium",
        },
        "allergies": "none",
        "conditions": "none",
        "diet": "vegetarian",
    },
    {
        "name": "Gulab Jamun (very high sugar) → should be AVOID",
        "expected": "avoid",
        "product": {
            "product_name": "Gulab Jamun",
            "nutriments": {
                "energy_kcal_100g": 387.0,
                "fat_100g": 13.5,
                "saturated_fat_100g": 6.5,
                "sugars_100g": 48.0,
                "salt_100g": 0.1,
                "proteins_100g": 5.5,
                "fiber_100g": 0.3,
            },
            "ingredients": "khoya, maida, sugar, ghee, cardamom, rose water",
            "source_type": "inferred",
            "data_confidence": "medium",
        },
        "allergies": "none",
        "conditions": "diabetes",
        "diet": "vegetarian",
    },
    {
        "name": "Paneer (lactose allergy) → should be AVOID",
        "expected": "avoid",
        "product": {
            "product_name": "Paneer (cottage cheese)",
            "nutriments": {
                "energy_kcal_100g": 265.0,
                "fat_100g": 20.8,
                "saturated_fat_100g": 13.0,
                "sugars_100g": 1.2,
                "salt_100g": 0.5,
                "proteins_100g": 18.3,
                "fiber_100g": 0.0,
            },
            "ingredients": "milk, citric acid",
            "source_type": "inferred",
            "data_confidence": "high",
        },
        "allergies": "lactose, milk",
        "conditions": "none",
        "diet": "vegetarian",
    },
    {
        "name": "Dal (lentil soup, high protein) → should be EAT",
        "expected": "eat",
        "product": {
            "product_name": "Dal (Masoor / Red Lentil Soup)",
            "nutriments": {
                "energy_kcal_100g": 116.0,
                "fat_100g": 1.8,
                "saturated_fat_100g": 0.3,
                "sugars_100g": 1.0,
                "salt_100g": 0.5,
                "proteins_100g": 9.0,
                "fiber_100g": 4.0,
            },
            "ingredients": "masoor dal, water, turmeric, cumin, onion, tomato, salt",
            "source_type": "inferred",
            "data_confidence": "medium",
        },
        "allergies": "none",
        "conditions": "none",
        "diet": "vegetarian",
    },
    {
        "name": "Samosa (deep fried, high fat) → should be AVOID/SOMETIMES",
        "expected": "avoid",
        "product": {
            "product_name": "Samosa (fried pastry)",
            "nutriments": {
                "energy_kcal_100g": 308.0,
                "fat_100g": 17.0,
                "saturated_fat_100g": 7.0,
                "sugars_100g": 2.0,
                "salt_100g": 0.8,
                "proteins_100g": 5.5,
                "fiber_100g": 1.5,
            },
            "ingredients": "maida, potato, peas, cumin, coriander, oil, salt",
            "source_type": "inferred",
            "data_confidence": "medium",
        },
        "allergies": "none",
        "conditions": "cholesterol",
        "diet": "vegetarian",
    },
    {
        "name": "Ragi Porridge (millet, high fiber) → should be EAT",
        "expected": "eat",
        "product": {
            "product_name": "Ragi Porridge (finger millet)",
            "nutriments": {
                "energy_kcal_100g": 82.0,
                "fat_100g": 1.3,
                "saturated_fat_100g": 0.2,
                "sugars_100g": 1.5,
                "salt_100g": 0.1,
                "proteins_100g": 3.5,
                "fiber_100g": 5.0,
            },
            "ingredients": "ragi flour, water, jaggery, cardamom",
            "source_type": "inferred",
            "data_confidence": "medium",
        },
        "allergies": "none",
        "conditions": "diabetes",
        "diet": "vegetarian",
    },
    {
        "name": "Chicken Biryani (non-veg for vegetarian) → should be AVOID",
        "expected": "avoid",
        "product": {
            "product_name": "Chicken Biryani",
            "nutriments": {
                "energy_kcal_100g": 175.0,
                "fat_100g": 6.5,
                "saturated_fat_100g": 2.0,
                "sugars_100g": 1.0,
                "salt_100g": 0.9,
                "proteins_100g": 12.0,
                "fiber_100g": 0.5,
            },
            "ingredients": "basmati rice, chicken, yogurt, onion, spices, ghee, saffron",
            "source_type": "inferred",
            "data_confidence": "medium",
        },
        "allergies": "none",
        "conditions": "none",
        "diet": "vegetarian",
    },
    {
        "name": "Jalebi (extreme sugar) → should be AVOID",
        "expected": "avoid",
        "product": {
            "product_name": "Jalebi",
            "nutriments": {
                "energy_kcal_100g": 450.0,
                "fat_100g": 12.0,
                "saturated_fat_100g": 5.0,
                "sugars_100g": 65.0,
                "salt_100g": 0.1,
                "proteins_100g": 3.0,
                "fiber_100g": 0.0,
            },
            "ingredients": "maida, sugar, ghee, saffron, citric acid",
            "source_type": "inferred",
            "data_confidence": "medium",
        },
        "allergies": "gluten",
        "conditions": "diabetes",
        "diet": "vegetarian",
    },
    {
        "name": "Sprouts Salad (high protein, low cal) → should be EAT",
        "expected": "eat",
        "product": {
            "product_name": "Sprouts Salad (moong bean)",
            "nutriments": {
                "energy_kcal_100g": 68.0,
                "fat_100g": 0.4,
                "saturated_fat_100g": 0.1,
                "sugars_100g": 1.0,
                "salt_100g": 0.2,
                "proteins_100g": 7.5,
                "fiber_100g": 3.8,
            },
            "ingredients": "moong sprouts, onion, tomato, lemon, chaat masala, coriander",
            "source_type": "inferred",
            "data_confidence": "medium",
        },
        "allergies": "none",
        "conditions": "none",
        "diet": "vegan",
    },
    {
        "name": "Chole Bhature (high cal + hypertension) → should be AVOID",
        "expected": "avoid",
        "product": {
            "product_name": "Chole Bhature",
            "nutriments": {
                "energy_kcal_100g": 320.0,
                "fat_100g": 18.0,
                "saturated_fat_100g": 4.5,
                "sugars_100g": 3.0,
                "salt_100g": 1.8,
                "proteins_100g": 8.0,
                "fiber_100g": 3.5,
            },
            "ingredients": "chickpeas, maida, oil, spices, salt, baking soda",
            "source_type": "inferred",
            "data_confidence": "medium",
        },
        "allergies": "gluten",
        "conditions": "hypertension",
        "diet": "vegetarian",
    },
]


def load_model(adapter_path: str, force_cpu: bool = False):
    """Load base model + LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # Determine device
    if force_cpu or not torch.cuda.is_available():
        device_map = "cpu"
        dtype = torch.float32
        device_label = "CPU"
    else:
        device_map = "auto"
        dtype = torch.float16
        device_label = f"GPU ({torch.cuda.get_device_name(0)})"

    print(f"🔧 Device: {device_label}")
    print(f"📦 Base model: {base_model_name}")
    print(f"🔗 Adapter: {adapter_path}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,  # Use tokenizer from adapter dir (has our chat template)
        trust_remote_code=True,
    )

    # Load base model
    print("Loading base model...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"  Base model loaded in {time.time() - t0:.1f}s")

    # Apply LoRA adapter
    print("Loading LoRA adapter...")
    t0 = time.time()
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print(f"  Adapter loaded in {time.time() - t0:.1f}s")

    return model, tokenizer


def run_inference(model, tokenizer, product: dict,
                  allergies: str = "none", conditions: str = "none",
                  diet: str = "none") -> tuple[dict, float]:
    """Run a single inference and return (parsed_result, latency_ms)."""
    import torch

    # Build prompt (same format as training data)
    nut = product.get("nutriments", {})

    def fmt(key):
        v = nut.get(key)
        return str(round(v, 1)) if v is not None else "not available"

    user_msg = f"""[PRODUCT]
Name: {product.get('product_name', 'Unknown')}
Source: {product.get('source_type', 'inferred')}
Data Confidence: {product.get('data_confidence', 'low')}
Nutrients (per 100g):
  Calories: {fmt('energy_kcal_100g')}
  Fat: {fmt('fat_100g')}
  Saturated Fat: {fmt('saturated_fat_100g')}
  Sugar: {fmt('sugars_100g')}
  Salt: {fmt('salt_100g')}
  Protein: {fmt('proteins_100g')}
  Fiber: {fmt('fiber_100g')}
Ingredients: {product.get('ingredients', 'not available') or 'not available'}

[USER PROFILE]
Allergies: {allergies}
Conditions: {conditions}
Diet: {diet}

[OUTPUT] Respond with ONLY valid JSON:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
            repetition_penalty=1.1,
        )
    latency_ms = (time.time() - t0) * 1000

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)

    # Parse JSON from output
    try:
        json_match = re.search(r'\{[^{}]+\}', raw)
        if json_match:
            result = json.loads(json_match.group())
            if all(k in result for k in ["verdict", "reason", "confidence"]):
                return result, latency_ms
    except (json.JSONDecodeError, ValueError):
        pass

    return {"raw": raw, "verdict": "parse_error", "reason": "Could not parse model output", "confidence": 0.0}, latency_ms


def print_result(test_name: str, expected: str, result: dict, latency_ms: float):
    """Pretty-print a test result."""
    verdict = result.get("verdict", "???")
    reason = result.get("reason", "")
    confidence = result.get("confidence", 0)

    # Check pass/fail
    match = verdict.lower() == expected.lower()
    status = "✅ PASS" if match else "❌ FAIL"

    print(f"\n{'─' * 60}")
    print(f"  {status}  {test_name}")
    print(f"{'─' * 60}")
    print(f"  Expected:   {expected}")
    print(f"  Got:        {verdict}  (confidence: {confidence})")
    print(f"  Reason:     {reason}")
    print(f"  Latency:    {latency_ms:.0f}ms")

    if "raw" in result:
        print(f"  Raw output: {result['raw'][:200]}")

    return match


def main():
    parser = argparse.ArgumentParser(description="NutriAssist local inference tester")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=str(DEFAULT_ADAPTER),
        help=f"Path to LoRA adapter directory (default: {DEFAULT_ADAPTER})",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (slower)")
    parser.add_argument(
        "--tests",
        type=str,
        default="all",
        help="Comma-separated test indices (1-based) or 'all' (e.g., '1,3,5')",
    )
    args = parser.parse_args()

    # Validate adapter path
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"❌ Adapter not found at: {adapter_path}")
        print(f"   Make sure you've downloaded the adapter to that location.")
        sys.exit(1)

    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        print(f"❌ No adapter_config.json found in: {adapter_path}")
        print(f"   This doesn't look like a valid LoRA adapter directory.")
        sys.exit(1)

    # Select test cases
    if args.tests == "all":
        selected = TEST_CASES
    else:
        indices = [int(i) - 1 for i in args.tests.split(",")]
        selected = [TEST_CASES[i] for i in indices if 0 <= i < len(TEST_CASES)]

    print("=" * 60)
    print("  🥗  NutriAssist Local Inference Test")
    print("=" * 60)
    print(f"  Adapter:    {adapter_path}")
    print(f"  Tests:      {len(selected)} cases")
    print(f"  CPU mode:   {args.cpu}")
    print("=" * 60)

    # Load model
    print("\n📥 Loading model...\n")
    t0 = time.time()
    model, tokenizer = load_model(str(adapter_path), force_cpu=args.cpu)
    load_time = time.time() - t0
    print(f"\n✅ Model ready in {load_time:.1f}s\n")

    # Run tests
    passed = 0
    total = len(selected)
    latencies = []

    for tc in selected:
        result, latency = run_inference(
            model, tokenizer,
            product=tc["product"],
            allergies=tc["allergies"],
            conditions=tc["conditions"],
            diet=tc["diet"],
        )
        match = print_result(tc["name"], tc["expected"], result, latency)
        if match:
            passed += 1
        latencies.append(latency)

    # Summary
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    print("\n" + "=" * 60)
    print(f"  📊  RESULTS: {passed}/{total} passed  ({passed/total*100:.0f}%)")
    print(f"  ⏱️  Avg latency: {avg_latency:.0f}ms")
    print(f"  ⏱️  Model load: {load_time:.1f}s")
    print("=" * 60)

    if passed < total:
        print(f"\n⚠️  {total - passed} test(s) failed. The model may need more training data")
        print("   or the test expectations need adjusting.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    main()
