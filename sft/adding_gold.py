import json

inp = "pretrain/bench_v1.jsonl"
out = "pretrain/bench_v1.gold.jsonl"

with open(inp, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        it = json.loads(line)
        if "gold" not in it and "answer" in it:
            it["gold"] = it["answer"]
        g.write(json.dumps(it, ensure_ascii=False) + "\n")

print("wrote:", out)
