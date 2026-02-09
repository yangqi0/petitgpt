import json, random
from tokenizers import Tokenizer

BOS_ID, EOS_ID = 2, 3
SYS_OPEN = "[SYS]\n"; SYS_CLOSE="\n[/SYS]\n"
USER_OPEN="[USER]\n"; USER_CLOSE="\n[/USER]\n"
ASSIST_OPEN="[ASSISTANT]\n"; ASSIST_CLOSE="\n[/ASSISTANT]\n"

def clean(s): return s.replace("\r\n","\n").replace("\r","\n").strip()

def render_segments(messages, default_system="You are a helpful assistant."):
    if messages and messages[0].get("role")!="system":
        messages=[{"role":"system","content":default_system}]+messages
    segs=[]
    i=0
    if messages and messages[0].get("role")=="system":
        t=clean(messages[0].get("content",""))
        if t:
            segs += [(SYS_OPEN,False),(t,False),(SYS_CLOSE,False)]
        i=1
    for m in messages[i:]:
        r=(m.get("role") or "").strip().lower()
        t=clean(m.get("content",""))
        if not t: continue
        if r=="user":
            segs += [(USER_OPEN,False),(t,False),(USER_CLOSE,False)]
        elif r=="assistant":
            segs += [(ASSIST_OPEN,False),(t,True),(ASSIST_CLOSE,False)]
    return segs

def encode_strip(tok, text):
    ids=tok.encode(text).ids
    if ids and ids[0]==BOS_ID: ids=ids[1:]
    if ids and ids[-1]==EOS_ID: ids=ids[:-1]
    return ids

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
path="datasets/sft/ultrachat.val.canon.jsonl"
lines=open(path,"r",encoding="utf-8").read().splitlines()
for _ in range(5):
    ex=json.loads(random.choice(lines))
    segs=render_segments(ex["messages"])
    # collect supervised raw text + decoded from ids
    sup_raw=[]
    sup_ids=[]
    for s, sup in segs:
        if sup:
            sup_raw.append(s)
            sup_ids += encode_strip(tok, s)
    raw="\n".join(sup_raw)[:300]
    dec=tok.decode(sup_ids)[:300]
    print("="*80)
    print("[raw supervised first300]:", raw)
    print("[dec supervised first300]:", dec)
