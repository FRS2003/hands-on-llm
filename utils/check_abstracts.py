import pickle
with open("data/vsd_rag_db/texts_en.pkl", "rb") as f:
    texts = pickle.load(f)
with open("data/vsd_rag_db/metas_en.pkl", "rb") as f:
    metas = pickle.load(f)
for pid in [12, 10, 33, 14, 17]:
    for i, m in enumerate(metas):
        if m["id"] == pid:
            print(f"=== [{pid}] {m['title'][:60]} ===")
            print(texts[i][:400])
            print()
            break
