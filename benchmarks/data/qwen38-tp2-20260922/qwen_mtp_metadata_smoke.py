"""Exercise Qwen MTP decode, mixed prefills, and structured responses at C4."""

import concurrent.futures
import json
import os

import requests


BASE = os.environ.get("QWEN_BASE", "http://192.168.0.115:5071")
SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}, "sum": {"type": "integer"}},
    "required": ["code", "sum"],
    "additionalProperties": False,
}


def check(index):
    code = f"AMBER-{701 + index}"
    expected = {"code": code, "sum": 13 + index}
    filler = "\n".join(
        f"Reference {line}: cedar trees are green and rivers flow downhill."
        for line in range((0, 10, 270, 1900)[index % 4])
    )
    body = {
        "model": "Qwen3.8-Flash-Next",
        "messages": [
            {"role": "system", "content": f"The access code is {code}.\n{filler}"},
            {"role": "user", "content": (
                f"Return one JSON object with code equal to the access code and "
                f"sum equal to {11 + index} plus 2. No other output."
            )},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        "reasoning_effort": "medium",
        "max_tokens": 1024,
    }
    if index >= 8:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "recall", "strict": True, "schema": SCHEMA},
        }
    client = requests.Session()
    client.trust_env = False
    response = client.post(BASE + "/v1/chat/completions", json=body, timeout=240)
    response.raise_for_status()
    output = response.json()
    message = output["choices"][0]["message"]
    content = message.get("content") or ""
    cleaned = content.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    try:
        parsed = json.loads(cleaned)
    except ValueError:
        parsed = None
    return {
        "index": index, "structured": index >= 8, "expected": expected,
        "passed": parsed == expected, "response": output,
    }


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(check, range(16)))
for result in results:
    print(json.dumps(result), flush=True)
print(json.dumps({"passed": sum(result["passed"] for result in results), "total": 16}))
raise SystemExit(0 if all(result["passed"] for result in results) else 1)
