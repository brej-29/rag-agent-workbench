"""Small in-memory BM25 over provisions: exact terms and article references dense retrieval misses."""
import math
import re
from collections import Counter
from typing import Iterable, List, Tuple

_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "by", "with", "is", "are", "be", "what",
         "does", "do", "which", "that", "this", "it", "as", "at", "from", "under", "shall", "how", "when"}
_ALIASES = {"article": "art", "articles": "art", "paragraph": "par", "recital": "rct", "annex": "anx"}


def tokenize(text: str) -> List[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        t = str(int(t)) if t.isdigit() and len(t) < 6 else _ALIASES.get(t, t)  # "AC-02" == "AC-2"
        if t not in _STOP:
            out.append(t)
    return out


class BM25:
    def __init__(self, docs: Iterable[Tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.ids, self.tfs, self.lens = [], [], []
        df: Counter = Counter()
        for doc_id, text in docs:
            toks = tokenize(text)
            tf = Counter(toks)
            self.ids.append(doc_id)
            self.tfs.append(tf)
            self.lens.append(len(toks))
            df.update(tf.keys())
        n = len(self.ids)
        self.avg = sum(self.lens) / n if n else 0.0
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    @classmethod
    def from_provisions(cls, provisions: Iterable[dict]) -> "BM25":
        return cls((p["id"], " ".join(x for x in (p.get("ref"), p.get("title"), p.get("text")) if x)) for p in provisions)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q = [t for t in tokenize(query) if t in self.idf]
        if not q:
            return []
        scores = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scores.append((self.ids[i], s))
        return sorted(scores, key=lambda x: -x[1])[:k]
