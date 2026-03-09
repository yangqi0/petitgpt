# pretrain/extract_smollm_python_edu.py
import argparse
import gzip
import json
import os

import boto3
from botocore.exceptions import ClientError

from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--num_proc", type=int, default=16)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--min_int_score", type=int, default=4)
    ap.add_argument("--min_chars", type=int, default=200)
    ap.add_argument("--max_chars", type=int, default=50000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "python-edu",
        split="train",
        num_proc=args.num_proc,
    )

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    s3 = boto3.client("s3")
    bucket_name = "softwareheritage"

    n_total = 0
    n_keep = 0
    n_fail = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for ex in ds:
            n_total += 1

            if int(ex.get("int_score", 0)) < args.min_int_score:
                continue

            blob_id = ex["blob_id"]
            key = f"content/{blob_id}"

            try:
                obj = s3.get_object(Bucket=bucket_name, Key=key)
                with gzip.GzipFile(fileobj=obj["Body"]) as fin:
                    text = fin.read().decode("utf-8", errors="ignore")
            except ClientError:
                n_fail += 1
                continue
            except Exception:
                n_fail += 1
                continue

            text = text.strip()
            if not text:
                continue
            if len(text) < args.min_chars or len(text) > args.max_chars:
                continue

            rec = {
                "text": text,
                "meta": {
                    "source": "smollm_python_edu",
                    "repo_name": ex.get("repo_name", ""),
                    "path": ex.get("path", ""),
                    "score": float(ex.get("score", 0.0)),
                    "int_score": int(ex.get("int_score", 0)),
                    "blob_id": blob_id,
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_keep += 1

            if n_keep % 1000 == 0:
                print(f"[keep={n_keep}] total={n_total} fail={n_fail}")

    print(
        {
            "total_seen": n_total,
            "kept": n_keep,
            "failed_download": n_fail,
            "out": args.out,
        }
    )


if __name__ == "__main__":
    main()
