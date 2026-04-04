"""Quick API test."""
import httpx
import json

base = "http://127.0.0.1:8000"

# Health
r = httpx.get(f"{base}/health")
print("Health:", r.json())

# Test 1: Idli
r = httpx.post(f"{base}/query", json={
    "text": "is idli healthy?",
    "user_profile": {"allergies": [], "conditions": ["diabetes"],
                     "diet_type": "vegetarian", "goal": "weight_loss"}
})
d = r.json()
print(f"\n--- is idli healthy? ---")
print(f"Product: {d['product_name']}")
print(f"Verdict: {json.dumps(d['verdict'], indent=2)}")
print(f"Response: {d['response_text']}")
print(f"Latency: {d['latency_ms']}ms")

# Test 2: Gulab Jamun
r = httpx.post(f"{base}/query", json={"text": "tell me about gulab jamun"})
d = r.json()
print(f"\n--- gulab jamun ---")
print(f"Product: {d['product_name']}")
print(f"Verdict: {json.dumps(d['verdict'], indent=2)}")
print(f"Response: {d['response_text']}")

# Test 3: Paneer + lactose allergy
r = httpx.post(f"{base}/query", json={
    "text": "is paneer good",
    "user_profile": {"allergies": ["lactose", "milk"], "conditions": [],
                     "diet_type": "", "goal": ""}
})
d = r.json()
print(f"\n--- paneer (lactose allergy) ---")
print(f"Product: {d['product_name']}")
print(f"Verdict: {json.dumps(d['verdict'], indent=2)}")
print(f"Response: {d['response_text']}")

# Test 4: Chai
r = httpx.post(f"{base}/query", json={"text": "is chai healthy?"})
d = r.json()
print(f"\n--- chai ---")
print(f"Product: {d['product_name']}")
print(f"Verdict: {json.dumps(d['verdict'], indent=2)}")
print(f"Latency: {d['latency_ms']}ms")

# Test 5: Casual
r = httpx.post(f"{base}/query", json={"text": "hello"})
d = r.json()
print(f"\n--- hello ---")
print(f"Intent: {d['intent']}")
print(f"Response: {d['response_text']}")
