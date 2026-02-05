from tokenizers import Tokenizer
tok = Tokenizer.from_file("../tokenizer/tokenizer.json")

print("id 2 ->", tok.id_to_token(2))
print("id 3 ->", tok.id_to_token(3))

# 如果你在训练 tokenizer 时用的是 [BOS]/[EOS] 这种字符串，也顺便检查：
for s in ["[BOS]","[EOS]","<BOS>","<EOS>","<|bos|>","<|eos|>"]:
    try:
        i = tok.token_to_id(s)
        if i is not None:
            print(f"{s} -> {i}")
    except Exception:
        pass
