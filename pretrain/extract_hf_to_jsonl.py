import argparse, json, os, random
from datasets import load_dataset
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset name, e.g. HuggingFaceFW/fineweb-edu")
    ap.add_argument("--config", default=None, help="HF config name (optional)")
    ap.add_argument("--split", default="train", help="split name")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_docs", type=int, default=0, help="stop after N docs (0=unlimited)")
    ap.add_argument("--text_field", default="text", help="field to use as text if present")
    ap.add_argument("--min_chars", type=int, default=200, help="drop very short samples")
    ap.add_argument("--data_dir", default=None, help="HF data_dir (subdirectory), e.g. python for starcoderdata")
    args = ap.parse_args()

    random.seed(args.seed)

    ds = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
        data_dir=args.data_dir,
        token=True,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in tqdm(ds, desc=f"stream {args.dataset}:{args.split}"):
            t = ex.get(args.text_field, None)
            if not isinstance(t, str):
                # fallback: try common fields
                for k in ("content", "article", "raw", "completion"):
                    v = ex.get(k, None)
                    if isinstance(v, str):
                        t = v
                        break
            if not isinstance(t, str):
                continue
            t = t.strip()
            if len(t) < args.min_chars:
                continue
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
            n += 1
            if args.max_docs and n >= args.max_docs:
                break

    print(f"wrote {n} docs -> {args.out}")

    # ---- clean shutdown to avoid rare C++ abort at interpreter exit ----
    try:
        # if tqdm object exists, close it (only if you used it explicitly)
        pass
    except Exception:
        pass

    try:
        # Explicitly drop streaming dataset iterator resources
        del ds
    except Exception:
        pass

    import gc
    gc.collect()

if __name__ == "__main__":
    main()
