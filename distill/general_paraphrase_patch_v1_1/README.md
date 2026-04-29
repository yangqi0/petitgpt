# general_paraphrase_templates_v1_1.py

This patched version fixes the overly strict template paraphrase validator.

Main changes:

1. `lost_quoted_text`
   - old: required quoted text to appear almost verbatim
   - new: fuzzy token-overlap preservation check

2. `lost_rewrite_contract`
   - old: required exact phrase `Return only the rewritten text`
   - new: accepts equivalent phrasings such as:
     - `Only return the rewritten text`
     - `Provide only the rewritten text`
     - `Do not include any explanation`

3. `lost_email_task`
   - old: required literal `email` or `message`
   - new: accepts related wording such as `note`, `follow-up`, `professional`, `polite`

4. Dedup is less aggressive:
   - lexical threshold: 0.82 -> 0.92
   - semantic threshold: 0.93 -> 0.965
   - cluster/parent cap: 2 -> 3

5. Adds optional:
   - `--out_all_jsonl`
   so you can inspect accepted and rejected paraphrases.

## Recommended command

```bash
python distill/general_paraphrase_templates_v1_1.py \
  --in_jsonl datasets/distill/general_template_seeds_v1.jsonl \
  --out_raw_jsonl datasets/distill/general_template_paraphrases_raw_v1.jsonl \
  --out_dedup_jsonl datasets/distill/general_template_paraphrases_dedup_v1.jsonl \
  --out_all_jsonl datasets/distill/general_template_paraphrases_all_v1.jsonl \
  --api_base http://127.0.0.1:8000/v1 \
  --model teacher \
  --temperature 0.5
```

Expected rough target:

- raw accepted: 180-320
- dedup accepted: 120-250
