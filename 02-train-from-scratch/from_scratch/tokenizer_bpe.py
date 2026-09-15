# -*- coding: utf-8 -*-
"""
from_scratch/tokenizer_bpe.py
=============================
不依赖任何第三方库、纯标准库手写的 **字节级 BPE（Byte-Pair Encoding）分词器**。

为什么是「字节级」
------------------
若以「词」为初始单位，词表永远列不完，遇到没见过的词只能返回 <unk>（OOV）。
GPT 系列的做法是先把文本编码成 UTF-8 字节，用 0..255 这 256 个单字节作为初始词表，
再在其上做 BPE 合并。这样**任何字符（中文、生僻字、emoji）都能被表示，从根本上消除 OOV**。

BPE 在做什么（一句话）
----------------------
训练时反复找到「相邻且共现频率最高」的一对符号，把它们合并成一个新符号加入词表；
编码时按训练得到的合并顺序，对新文本做同样的合并。常见拼写因此被拼成更少的 token。

本文件包含
----------
- BPETokenizer.train : 从语料学习 merge 规则与词表
- encode / decode    : 文本 <-> token id（保证可逆）
- save / load        : 只存 merge 规则即可重建词表
运行 ``python tokenizer_bpe.py`` 可看到一个中英混合的小演示。
"""
import re
import json
from collections import Counter
from typing import Dict, List, Tuple, Iterable

# 256 个单字节是永远存在的「底座」词表；特殊 token 紧随其后
DEFAULT_SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

# 预分词（pre-tokenization）：BPE 只在「词块内部」合并，绝不跨过词边界。
#   1) 可选前导空白 + 连续英文/数字   2) 可选前导空白 + 单个其它字符(中文/标点逐字)
#   3) 残余的连续空白
# 把前导空格挂在词上，是 GPT 的常见技巧，避免 token 跨越空格语义。
WORD_RE = re.compile(r"\s?[A-Za-z0-9]+|\s?[^\sA-Za-z0-9]|\s+")


def _pretokenize(text: str) -> List[str]:
    """把整句切成若干「词块」，BPE 的合并只发生在每个词块内部。"""
    return WORD_RE.findall(text)


def _merge_pair(ids: Tuple[int, ...], a: int, b: int, new: int) -> Tuple[int, ...]:
    """在一个词块的符号序列里，把所有相邻的 (a,b) 原地替换成新符号 new。"""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[int, int]],
                 special: List[str] = None):
        self.vocab = dict(vocab)                       # id -> bytes
        self.merges = [tuple(m) for m in merges]       # 有序合并规则 (a,b)
        self.special_names = list(special or [])
        self.n_special = len(self.special_names)
        self.base = 256 + self.n_special               # 第一个 merge 新 token 的 id
        self.token_to_id = {name: 256 + j for j, name in enumerate(self.special_names)}
        # (a,b) -> 它是第几条 merge（编码时按「最早学到的优先」合并）
        self.merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        self.unk_id = self.token_to_id.get("<unk>")

    # ---------- 便捷属性 ----------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self): return self.token_to_id.get("<pad>")
    @property
    def bos_id(self): return self.token_to_id.get("<bos>")
    @property
    def eos_id(self): return self.token_to_id.get("<eos>")

    # ---------- 训练 ----------
    @staticmethod
    def _build_vocab(special: List[str], merges: List[Tuple[int, int]]) -> Dict[int, bytes]:
        vocab = {i: bytes([i]) for i in range(256)}                 # 单字节底座
        for j, name in enumerate(special):                         # 特殊 token
            vocab[256 + j] = name.encode("utf-8")
        base = 256 + len(special)
        for k, (a, b) in enumerate(merges):                        # 每次合并=两子串拼接
            vocab[base + k] = vocab[a] + vocab[b]
        return vocab

    @classmethod
    def _build_from_counts(cls, word_counts: Dict[Tuple[int, ...], int],
                           vocab_size: int, special: List[str]):
        """核心训练循环：输入 {词块字节序列: 词频}，逐轮合并最高频相邻对。"""
        base = 256 + len(special)
        merges: List[Tuple[int, int]] = []
        vocab = cls._build_vocab(special, merges)
        cur = dict(word_counts)
        n_merges = max(0, vocab_size - len(vocab))
        for _ in range(n_merges):
            pair_counts = Counter()
            for word, freq in cur.items():                         # 频次按词频加权
                for k in range(len(word) - 1):
                    pair_counts[(word[k], word[k + 1])] += freq
            if not pair_counts:
                break                                              # 没有可合并对，提前结束
            # 频次最高者优先；频次相同按符号对升序，保证结果可复现
            pair = sorted(pair_counts, key=lambda p: (-pair_counts[p], p))[0]
            a, b = pair
            new_id = base + len(merges)
            vocab[new_id] = vocab[a] + vocab[b]
            merges.append(pair)
            cur = {_merge_pair(w, a, b, new_id): f for w, f in cur.items()}
        return cls(vocab, merges, special)

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 300,
              special: List[str] = None) -> "BPETokenizer":
        """从原始语料出发：先预分词、统计词频，再学习 BPE 合并。"""
        special = list(DEFAULT_SPECIAL if special is None else special)
        word_counts = Counter()
        for text in texts:
            for word in _pretokenize(text):
                word_counts[tuple(word.encode("utf-8"))] += 1
        return cls._build_from_counts(word_counts, vocab_size, special)

    # ---------- 编码 ----------
    def _bpe_word(self, syms: Tuple[int, ...]) -> List[int]:
        ids = list(syms)
        while len(ids) >= 2:
            best_i, best_rank = None, None
            for i in range(len(ids) - 1):                         # 选「最早学到」的可合并对
                rank = self.merge_rank.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_i, best_rank = i, rank
            if best_i is None:
                break
            a, b = ids[best_i], ids[best_i + 1]
            ids = ids[:best_i] + [self.base + best_rank] + ids[best_i + 2:]
        return ids

    def encode(self, text: str) -> List[int]:
        """文本 -> token id 序列。因为底座是全部 256 个字节，所以永不 OOV。"""
        ids: List[int] = []
        for word in _pretokenize(text):
            ids.extend(self._bpe_word(tuple(word.encode("utf-8"))))
        return ids

    # ---------- 解码 ----------
    def decode(self, ids: Iterable[int]) -> str:
        """token id -> 文本；逐 id 取出字节片段拼接后整体 UTF-8 解码（保证多字节中文不乱码）。"""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ---------- 持久化：只需存 special 与有序 merges，词表可确定性重建 ----------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"special": self.special_names,
                       "merges": [list(m) for m in self.merges]}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        special = d["special"]
        merges = [tuple(m) for m in d["merges"]]
        return cls(cls._build_vocab(special, merges), merges, special)


if __name__ == "__main__":
    corpus = [
        "the cat sat on the mat", "the dog ran on the mat",
        "tokenizer tokenizes tokens", "lower lowest lower",
        "手写一个字节级 BPE 分词器，中文 English 123 都能编码。",
    ]
    tok = BPETokenizer.train(corpus, vocab_size=300)
    print("词表大小:", tok.vocab_size, " 学到 merge 数:", len(tok.merges))
    print("前 8 条 merge:", [(chr(a) if a < 256 else a, chr(b) if b < 256 else b)
                            for a, b in tok.merges[:8]])
    for s in ["the mat", "tokenizer tokens", "手写 BPE 分词器，123！", "从未见过的新词 xyz"]:
        enc = tok.encode(s)
        print(f"\n原文: {s}\n ids: {enc}\n 还原: {tok.decode(enc)}  可逆={tok.decode(enc)==s}")
        print(f" 原始 UTF-8 字节数={len(s.encode('utf-8'))} -> BPE token 数={len(enc)}")