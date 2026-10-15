from __future__ import annotations

import os
from typing import Iterator, Optional

import numpy as np
import pandas as pd

WORDS_PER_PAGE = 512
DATA_WORDS = 508
ABSENT = -1e29

def _layout(raw: bytes):
    for order in ("<", ">"):
        d = np.frombuffer(raw[:WORDS_PER_PAGE * 8], dtype=order + "f8")
        if len(d) > 24 and abs(d[24] - 456789.0) < 1e-6:
            return 8, order

    for order in ("<", ">"):
        f = np.frombuffer(raw[:WORDS_PER_PAGE * 4], dtype=order + "f4")
        nwords, npages, nlast = f[25], f[26], f[27]
        if (0 < nwords < 5000 and float(nwords).is_integer()
                and 1 <= npages < 1e9 and float(npages).is_integer()
                and 0 <= nlast <= DATA_WORDS):
            return 4, order
    raise ValueError("Not a Datamine .dm file (header not recognised)")

def _text(words: np.ndarray, ws: int) -> str:
    b = words.tobytes()
    if ws == 8:
        b = b"".join(b[i:i + 4] for i in range(0, len(b), 8))
    return b.decode("latin-1").strip()

def read_dm_header(path: str) -> list[str]:
    with open(path, "rb") as fh:
        return list(_fields(fh.read(WORDS_PER_PAGE * 8))[0].keys())

def _fields(raw: bytes):
    ws, order = _layout(raw)
    ftype = order + ("f8" if ws == 8 else "f4")
    hdr_bytes = np.frombuffer(raw[:WORDS_PER_PAGE * ws], dtype=f"V{ws}")
    hdr_num = np.frombuffer(raw[:WORDS_PER_PAGE * ws], dtype=ftype)
    nwords = int(hdr_num[25])
    if 28 + 7 * nwords > WORDS_PER_PAGE:
        raise ValueError(f"This .dm file has {nwords} field words, more than one "
                         "header page holds; save it as CSV from Studio")
    fields: dict[str, dict] = {}
    for v in range(nwords):
        w = 28 + 7 * v
        name = _text(hdr_bytes[w:w + 2], ws)
        typ = _text(hdr_bytes[w + 2:w + 3], ws)[:1]
        pos = int(hdr_num[w + 3])
        wordno = int(hdr_num[w + 4])
        f = fields.setdefault(name, {"type": typ, "words": [], "default": None})
        f["words"].append((wordno, pos))
        if typ == "N":
            f["default"] = float(hdr_num[w + 6])
        else:
            f.setdefault("default_parts", []).append(
                (wordno, _text(hdr_bytes[w + 6:w + 7], ws)))
    rec_len = sum(1 for f in fields.values() for _, pos in f["words"] if pos > 0)
    return fields, ws, ftype, rec_len, int(hdr_num[26]), int(hdr_num[27])

def read_dm(path: str) -> pd.DataFrame:
    return next(iter_dm(path, rows=None))

def iter_dm(path: str, rows: Optional[int] = 1_000_000,
            columns: Optional[list] = None) -> Iterator[pd.DataFrame]:
    with open(path, "rb") as fh:
        fields, ws, ftype, rec_len, npages, nlast = _fields(
            fh.read(WORDS_PER_PAGE * 8))
        if columns is not None:
            fields = {k: v for k, v in fields.items() if k in set(columns)}
        page = WORDS_PER_PAGE * ws
        data_pages = npages - 1
        if rec_len == 0 or data_pages <= 0:
            yield _decode(fields, np.zeros((0, max(rec_len, 1))),
                          np.zeros((0, max(rec_len, 1)), dtype=f"V{ws}"), 0, ws)
            return
        per_page = DATA_WORDS // rec_len
        n = (data_pages - 1) * per_page + nlast
        if os.path.getsize(path) < page * npages:
            raise ValueError("The .dm file is shorter than its header says; "
                             "it may be truncated")
        step = data_pages if rows is None else max(1, rows // per_page)
        used = per_page * rec_len
        done = 0
        for p0 in range(0, data_pages, step):
            k = min(step, data_pages - p0)
            fh.seek(page * (1 + p0))
            body = fh.read(page * k)
            m = min(k * per_page, n - done)
            recs_num = np.frombuffer(body, dtype=ftype).reshape(k, WORDS_PER_PAGE)[
                :, :used].reshape(-1, rec_len)[:m]
            recs_raw = np.frombuffer(body, dtype=f"V{ws}").reshape(k, WORDS_PER_PAGE)[
                :, :used].reshape(-1, rec_len)[:m]
            yield _decode(fields, recs_num, recs_raw, m, ws)
            done += m

def _decode(fields: dict, recs_num, recs_raw, n: int, ws: int) -> pd.DataFrame:
    cols = {}
    for name, f in fields.items():
        words = sorted(f["words"])
        if f["type"] == "N":
            stored = [p for _, p in words if p > 0]
            if stored:
                vals = recs_num[:, stored[0] - 1].astype(float)
            else:
                vals = np.full(n, f["default"], dtype=float)
            cols[name] = np.where(vals <= ABSENT, np.nan, vals)
        else:
            if all(pos > 0 for _, pos in words):
                idx = [pos - 1 for _, pos in words]
                chunk = np.ascontiguousarray(recs_raw[:, idx]).view(np.uint8)
                chunk = np.ascontiguousarray(
                    chunk.reshape(n, len(idx), ws)[:, :, :4].reshape(n, -1))
                text = chunk.view(f"S{4 * len(idx)}").ravel()
                codes, uniq = pd.factorize(text)
                labels = np.array([u.decode("latin-1").strip() for u in uniq] + [""],
                                  dtype=object)
                cols[name] = labels[codes]
            else:
                default = "".join(t for _, t in sorted(f["default_parts"]))
                cols[name] = [default] * n
    return pd.DataFrame(cols)

class DmWriter:

    def __init__(self, path: str, fields: list, double: bool = True,
                 implicit: Optional[dict] = None):
        self.path = path
        self.ws = 8 if double else 4
        self.double = double
        self.ftype = "<f8" if double else "<f4"
        self.fields = [(str(n), int(np.ceil(w / 4)) * 4 if w else 0) for n, w in fields]
        for name, _ in self.fields + [(n, 0) for n in (implicit or {})]:
            if len(name) > 8:
                raise ValueError(f"Datamine field names are at most 8 characters: {name}")
        words = []
        pos = 0
        for name, width in self.fields:
            if width == 0:
                pos += 1
                words.append((name, "N", 1, pos, 0.0))
            else:
                for k in range(width // 4):
                    pos += 1
                    words.append((name, "A", k + 1, pos, ""))
        for name, v in (implicit or {}).items():
            words.append((name, "N", 1, 0, float(v)))
        if 28 + 7 * len(words) > WORDS_PER_PAGE:
            raise ValueError(f"{len(words)} field words are more than a .dm header "
                             "page holds (69); export fewer or narrower fields")
        if pos == 0:
            raise ValueError("A .dm table needs at least one stored field")
        self.words = words
        self.rec_len = pos
        self.per_page = DATA_WORDS // pos
        self.page_bytes = WORDS_PER_PAGE * self.ws
        self.rec_bytes = pos * self.ws
        self.n = 0
        self.pages = 0
        self._pending = b""
        self._fh = open(path, "wb")
        self._fh.write(b"\0" * self.page_bytes)

    def _text_word(self, s: str) -> bytes:
        b = s.encode("latin-1", "replace")[:4].ljust(4)
        return b + b"    " if self.double else b

    def _num_word(self, x: float) -> bytes:
        return np.array([x], dtype=self.ftype).tobytes()

    def _records(self, df: pd.DataFrame) -> bytes:
        n = len(df)
        parts = []
        for name, width in self.fields:
            col = df[name]
            if width == 0:
                v = pd.to_numeric(col, errors="coerce").to_numpy(float)
                v = np.where(np.isfinite(v), v, -1e30).astype(self.ftype)
                parts.append(v.view(np.uint8).reshape(n, self.ws))
            else:
                t = col.astype(object).where(col.notna(), "").astype(str)
                b = np.array([s.encode("latin-1", "replace") for s in t],
                             dtype=f"S{width}")

                u = b.view(np.uint8).reshape(n, width).copy()
                lens = np.char.str_len(b)
                u[np.arange(width)[None, :] >= lens[:, None]] = ord(" ")
                u = u.reshape(n, width // 4, 4)
                if self.double:
                    u = np.concatenate([u, np.full_like(u, ord(" "))], axis=2)
                parts.append(u.reshape(n, -1))
        return np.ascontiguousarray(np.concatenate(parts, axis=1)).tobytes()

    def write(self, df: pd.DataFrame) -> None:
        if not len(df):
            return
        buf = self._pending + self._records(df)
        self.n += len(df)
        full = len(buf) // (self.per_page * self.rec_bytes)
        used = self.per_page * self.rec_bytes
        for p in range(full):
            self._fh.write(buf[p * used:(p + 1) * used].ljust(self.page_bytes, b"\0"))
        self.pages += full
        self._pending = buf[full * used:]

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            nlast = self.per_page
            if self._pending or self.pages == 0:
                self._fh.write(self._pending.ljust(self.page_bytes, b"\0"))
                nlast = len(self._pending) // self.rec_bytes
                self.pages += 1
            hdr = [self._num_word(0.0)] * WORDS_PER_PAGE
            name8 = os.path.splitext(os.path.basename(self.path))[0].upper()[:8].ljust(8)
            hdr[0], hdr[1] = self._text_word(name8[:4]), self._text_word(name8[4:])
            for i in range(2, 22):
                hdr[i] = self._text_word("")
            hdr[24] = self._num_word(456789.0 if self.double else 20260928.0)
            hdr[25] = self._num_word(len(self.words))
            hdr[26] = self._num_word(self.pages + 1)
            hdr[27] = self._num_word(nlast)
            for v, (name, typ, wno, p, default) in enumerate(self.words):
                w = 28 + 7 * v
                nm = name.ljust(8)[:8]
                hdr[w], hdr[w + 1] = self._text_word(nm[:4]), self._text_word(nm[4:])
                hdr[w + 2] = self._text_word(typ)
                hdr[w + 3] = self._num_word(p)
                hdr[w + 4] = self._num_word(wno)
                hdr[w + 5] = self._text_word("")
                hdr[w + 6] = (self._num_word(default) if typ == "N"
                              else self._text_word(default))
            self._fh.seek(0)
            self._fh.write(b"".join(hdr))
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

def text_width(values) -> int:
    s = pd.Series(values, dtype=object)
    s = s.where(s.notna(), "").astype(str)
    return max(4, int(s.str.len().max() or 0)) if len(s) else 4

def write_dm(df: pd.DataFrame, path: str, double: bool = False,
             implicit: dict | None = None) -> None:
    fields = [(c, 0 if df[c].dtype.kind in "fiub" else text_width(df[c]))
              for c in df.columns]
    with DmWriter(path, fields, double=double, implicit=implicit) as w:
        w.write(df)
