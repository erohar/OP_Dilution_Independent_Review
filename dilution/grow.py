from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WASTE = -1
VOID = -2
FREE = -3

@dataclass
class Growth:
    up: np.ndarray
    dn: np.ndarray
    e_up: np.ndarray
    e_dn: np.ndarray

    @classmethod
    def none(cls, n: int) -> "Growth":
        z = np.zeros(n, dtype=np.int64)
        return cls(z.copy(), z.copy(), z.copy(), z.copy())

def allowance(lab: np.ndarray, ny: int, void_free: bool) -> tuple[np.ndarray, np.ndarray]:
    R, K = lab.shape
    lo = np.zeros((R, K), dtype=np.int64)
    hi = np.zeros((R, K), dtype=np.int64)
    w = np.zeros((K, R + 2), dtype=np.int8)
    w[:, 1:-1] = (lab == WASTE).T
    d = np.diff(w, axis=1)
    kc, r0 = np.nonzero(d == 1)
    kc2, r1 = np.nonzero(d == -1)
    if not len(kc):
        return lo, hi
    if not np.array_equal(kc, kc2):
        raise RuntimeError("Waste runs along Y failed to pair")
    n = r1 - r0
    below = lab[np.maximum(r0 - 1, 0), kc]
    above = lab[np.minimum(r1, R - 1), kc]
    ore_b, ore_a = below >= 0, above >= 0

    def free(t):
        return (t == FREE) | ((t == VOID) if void_free else np.zeros(len(t), bool))

    spare = np.maximum(n - ny, 0)
    a_lo = np.where(ore_b & ore_a, (spare + 1) // 2,
                    np.where(ore_b, np.where(free(above), n, spare), 0))
    a_hi = np.where(ore_b & ore_a, spare // 2,
                    np.where(ore_a, np.where(free(below), n, spare), 0))
    lo[r0, kc] = np.minimum(a_lo, ny)
    hi[r1 - 1, kc] = np.minimum(a_hi, ny)
    return lo, hi

def row_ok(lab: np.ndarray, r: int, a: int, b: int, m: int, void_free: bool) -> bool:
    row = lab[r]
    if not (row[a:b] == WASTE).all():
        return False
    for side in (row[max(a - m, 0):a][::-1], row[b:b + m]):
        hit = np.flatnonzero(side != WASTE)
        if not len(hit):
            continue
        i, c = int(hit[0]), int(side[hit[0]])
        if i == 0 or c == FREE or (c == VOID and void_free):
            continue
        return False
    return True

def _prefix(lab, rows, a, b, m, void_free) -> int:
    for i, r in enumerate(rows):
        if not row_ok(lab, r, a, b, m, void_free):
            return i
    return len(rows)

def _better(c1, v1, c0, v0, tc, tv, rule) -> bool:
    if rule == "recover":
        return c1 > c0 + tc or (c1 >= c0 - tc and v1 > v0 + tv)
    return v1 > v0 + tv

def _choose(val, cap, rows, a, b, v_block, rule) -> int:
    if not len(rows):
        return 0
    rv = val[rows, a:b].sum(axis=1)
    rc = cap[rows, a:b].sum(axis=1)
    cv, cc = np.cumsum(rv), np.cumsum(rc)
    tv = 1e-9 * (1.0 + abs(v_block) + float(np.abs(rv).sum()))
    tc = 1e-9 * (1.0 + float(np.abs(rc).sum()))
    best, bc, bv = 0, 0.0, 0.0
    for t in range(1, len(rows) + 1):
        if v_block + cv[t - 1] < -tv:
            continue
        if _better(cc[t - 1], cv[t - 1], bc, bv, tc, tv, rule):
            best, bc, bv = t, cc[t - 1], cv[t - 1]
    return best

def grow_blocks(lab: np.ndarray, val: np.ndarray, cap: np.ndarray,
                r0: np.ndarray, fa: np.ndarray, fb: np.ndarray,
                da: np.ndarray, db: np.ndarray, ny: int, m: int, ey: int,
                rule: str) -> Growth:
    n = len(r0)
    g = Growth.none(n)
    if not n or (ny <= 1 and ey <= 0):
        return g
    order = np.lexsort((fa, r0))
    v_block = np.array([val[r0[i]:r0[i] + ny, da[i]:db[i]].sum() for i in range(n)])
    top = r0 + ny
    if ny > 1:
        g_lo, g_hi = allowance(lab, ny, void_free=False)
        for i in order:
            a, b = fa[i], fb[i]
            t = min(ny - 1, int(g_lo[top[i], a:b].min()))
            rows = np.arange(top[i], top[i] + t)
            rows = rows[:_prefix(lab, rows, a, b, m, False)]
            t = _choose(val, cap, rows, da[i], db[i], v_block[i], rule)
            if t:
                lab[top[i]:top[i] + t, a:b] = i
                v_block[i] += val[top[i]:top[i] + t, da[i]:db[i]].sum()
                g.up[i] = t
        for i in order:
            a, b = fa[i], fb[i]
            t = min(ny - 1 - g.up[i], int(g_hi[r0[i] - 1, a:b].min()))
            rows = np.arange(r0[i] - 1, r0[i] - 1 - t, -1)
            rows = rows[:_prefix(lab, rows, a, b, m, False)]
            t = _choose(val, cap, rows, da[i], db[i], v_block[i], rule)
            if t:
                lab[r0[i] - t:r0[i], a:b] = i
                g.dn[i] = t
    if ey > 0:
        e_lo, e_hi = allowance(_design_lab(lab, r0, fa, fb, ny, g), ny, void_free=True)
        for i in order:
            a, b = fa[i], fb[i]
            e = min(ey, int(e_lo[top[i], a:b].min()) - g.up[i])
            start = top[i] + g.up[i]
            rows = np.arange(start, start + max(e, 0))
            e = _prefix(lab, rows, a, b, m, True)
            if e:
                lab[start:start + e, a:b] = i
                g.e_up[i] = e
        for i in order:
            a, b = fa[i], fb[i]
            e = min(ey, int(e_hi[r0[i] - 1, a:b].min()) - g.dn[i])
            start = r0[i] - 1 - g.dn[i]
            rows = np.arange(start, start - max(e, 0), -1)
            e = _prefix(lab, rows, a, b, m, True)
            if e:
                lab[start - e + 1:start + 1, a:b] = i
                g.e_dn[i] = e
    return g

def _design_lab(lab, r0, fa, fb, ny, g) -> np.ndarray:
    out = lab.copy()
    for i in range(len(r0)):
        if g.up[i]:
            out[r0[i] + ny:r0[i] + ny + g.up[i], fa[i]:fb[i]] = WASTE
        if g.dn[i]:
            out[r0[i] - g.dn[i]:r0[i], fa[i]:fb[i]] = WASTE
    return out

def narrow_waste(lab: np.ndarray, m: int, ny: int, rows: np.ndarray) -> int:
    count = 0
    for axis, need in ((1, m), (0, ny)):
        a = lab if axis == 1 else lab.T
        w = np.zeros((a.shape[0], a.shape[1] + 2), dtype=np.int8)
        w[:, 1:-1] = a == WASTE
        d = np.diff(w, axis=1)
        line, s = np.nonzero(d == 1)
        _, e = np.nonzero(d == -1)
        if not len(line):
            continue
        before = a[line, np.maximum(s - 1, 0)]
        after = a[line, np.minimum(e, a.shape[1] - 1)]
        bad = (s > 0) & (e < a.shape[1]) & (before >= 0) & (after >= 0) & (e - s < need)
        where = line if axis == 1 else e
        count += int((bad & np.isin(where, rows)).sum())
    return count
