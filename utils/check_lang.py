import pickle, re
with open("data/vsd_rag_db/texts_m3.pkl", "rb") as f:
    texts = pickle.load(f)
print(f"Total chunks: {len(texts)}")
cn = sum(1 for t in texts if re.search(r"[一-鿿]", t))
en = len(texts) - cn
print(f"Chinese: {cn}, English: {en}")
print()
for i in range(5):
    has_cn = "[CN]" if re.search(r"[一-鿿]", texts[i]) else "[EN]"
    print(f"[{i}] {has_cn} {texts[i][:120]}...")
