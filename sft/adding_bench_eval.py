import json

inp = "pretrain/bench_v1.gold.jsonl"
out = "pretrain/bench_v1.eval.jsonl"

def split_tests(t):
    if t is None:
        return None
    if isinstance(t, list):
        return [x.strip() for x in t if str(x).strip()]
    if isinstance(t, str):
        return [x.strip() for x in t.splitlines() if x.strip()]
    return None

with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        line=line.strip()
        if not line:
            continue
        it = json.loads(line)
        if it.get("task") == "code":
            it["tests"] = split_tests(it.get("tests"))
        g.write(json.dumps(it, ensure_ascii=False) + "\n")

print("wrote:", out)
