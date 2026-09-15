# -*- coding: utf-8 -*-
"""from_scratch/test_tokenizer_bpe.py —— 纯标准库、CPU 验证手写 BPE 分词器正确性。
运行: python test_tokenizer_bpe.py
"""
import os
import tempfile
from tokenizer_bpe import BPETokenizer, DEFAULT_SPECIAL

PASS = 0
def ok(name, cond):
    assert cond, f"FAIL: {name}"
    global PASS; PASS += 1; print(f"[PASS] {name}")

CORPUS = [
    "the cat sat on the mat", "the dog ran on the mat",
    "tokenizer tokenizes tokens", "lower lowest lower",
    "a quick brown fox jumps over the lazy dog",
    "hand written byte pair encoding is simple and robust",
    "手写一个字节级 BPE 分词器，中文与 English 123 混合都要可逆。",
]
tok = BPETokenizer.train(CORPUS, vocab_size=320)

# 1) 编解码可逆：英文/中文/标点/空白/数字
def test_roundtrip():
    for s in ["the mat", "brown fox 123", "手写 BPE 分词器，123！",
              "  leading and trailing  ", "混合 mixed 。\n换行"]:
        ok(f"可逆 roundtrip: {s[:12]!r}", tok.decode(tok.encode(s)) == s)

# 2) 词表不超过上限；底座 256 + 4 special 恒在
def test_vocab():
    ok("词表不超过设定上限", tok.vocab_size <= 320)
    ok("256 个单字节底座完整", all(tok.vocab[i] == bytes([i]) for i in range(256)))
    ok("特殊 token id 位置正确",
       tok.pad_id == 256 and tok.bos_id == 257 and tok.eos_id == 258 and tok.unk_id == 259)
    ok("特殊 token 可解码", tok.decode([tok.eos_id]) == "<eos>")

# 3) 第一条 merge 必须是语料里最高频的相邻对（用受控词频精确验证）
def test_first_merge():
    wc = {(97, 98): 4, (97, 99): 2}          # "ab"x4 vs "ac"x2
    t = BPETokenizer._build_from_counts(wc, 300, DEFAULT_SPECIAL)
    ok("最高频相邻对最先合并", t.merges[0] == (97, 98))

# 4) 压缩性：训练语料的 BPE token 数应少于原始 UTF-8 字节数
def test_compression():
    text = " ".join(CORPUS)
    ok("BPE token 数 < UTF-8 字节数", len(tok.encode(text)) < len(text.encode("utf-8")))

# 5) 无 OOV：训练时没出现过的字符/新词也能编码且无损还原
def test_no_oov():
    unseen = "zzz qwerty 生僻龘靐 emoji😀"
    enc = tok.encode(unseen)
    ok("未见文本可编码", isinstance(enc, list) and len(enc) > 0)
    ok("未见文本无损还原（字节回退无 OOV）", tok.decode(enc) == unseen)

# 6) 每条 merge 都满足 vocab[new] == vocab[a] + vocab[b]
def test_merge_consistency():
    good = all(tok.vocab[tok.base + k] == tok.vocab[a] + tok.vocab[b]
               for k, (a, b) in enumerate(tok.merges))
    ok("merge 词表=两子符号字节拼接", good)

# 7) 确定性：同样输入两次训练，merge 完全一致
def test_deterministic():
    t2 = BPETokenizer.train(CORPUS, vocab_size=320)
    ok("训练结果确定可复现", t2.merges == tok.merges)

# 8) save/load 后编码结果一致（词表可由 merges 重建）
def test_save_load():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bpe.json")
        tok.save(p)
        loaded = BPETokenizer.load(p)
        s = "the tokenizer 分词"
        ok("save/load 后编码一致", loaded.encode(s) == tok.encode(s))
        ok("save/load 后可逆", loaded.decode(loaded.encode(s)) == s)

# 9) 边界：空串与单字符
def test_edge():
    ok("空串编码为空列表", tok.encode("") == [])
    ok("空 id 解码为空串", tok.decode([]) == "")
    ok("单字符可逆", tok.decode(tok.encode("A")) == "A")

if __name__ == "__main__":
    test_roundtrip(); test_vocab(); test_first_merge(); test_compression()
    test_no_oov(); test_merge_consistency(); test_deterministic()
    test_save_load(); test_edge()
    print("-" * 60)
    print(f"BPE tokenizer self-test: {PASS} passed, 0 failed")
    raise SystemExit(0)