import httpx, time

tests = [
    ("is idli healthy", {"allergies": [], "conditions": [], "diet_type": "vegetarian"}),
    ("is kitkat healthy", {"allergies": [], "conditions": ["diabetes"], "diet_type": ""}),
    ("is gulab jamun healthy", {"allergies": [], "conditions": [], "diet_type": ""}),
    ("is paneer good", {"allergies": ["lactose"], "conditions": [], "diet_type": "vegetarian"}),
    ("is dosa healthy", {"allergies": [], "conditions": ["diabetes"], "diet_type": "vegetarian"}),
]

for text, profile in tests:
    t = time.time()
    r = httpx.post(
        "http://127.0.0.1:8000/query",
        json={"text": text, "user_profile": profile},
        timeout=30,
    )
    ms = (time.time() - t) * 1000
    d = r.json()
    v = d.get("verdict", {})
    vv = v.get("verdict", "?")
    print(f"\n--- {text} ---")
    print(f"  Verdict: {vv.upper()} ({v.get('confidence', 0):.0%})")
    print(f"  Reason:  {v.get('reason', '')}")
    print(f"  Response: {d['response_text']}")
    print(f"  Latency: {ms:.0f}ms")
