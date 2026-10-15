from __future__ import annotations

import datetime as _dt
import itertools
import shutil
import sys
import weakref
from collections import deque
import multiprocessing
import json
import logging
import math
from fractions import Fraction
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .merge import RULES as MERGE_RULES
from .grow import FREE, VOID, Growth, grow_blocks, narrow_waste
from .regions import DEFAULT_NAME, Region, as_regions, order_regions
from .regions import owner as region_owner
from .store import (CHUNK_ROWS, BlockStore, _ext, _no_dmx, as_store,
                    iter_chunks, make_work_dir, read_header)

log = logging.getLogger("dilution")

GRAMS_PER_OZ = 31.1034768
_EPS = 1e-9
_TOL = 1e-6

ORE_RULES = {
    "value": "most metal above cut-off",
    "recover": "keep above cut-off SMUs",
}
OUTSIDE = {
    "model": "diluted along the model X axis",
    "waste": "left as waste",
}
ELOS_AXES = {
    "xy": "X and Y",
    "x": "X only (legacy)",
}

@dataclass
class DilutionParams:
    input_path: str
    output_dir: str

    x_col: str = "XC"
    y_col: str = "YC"
    z_col: str = "ZC"
    grade_col: str = "AU"
    rescat_col: str = "RESCAT"
    density_col: Optional[str] = "DENSITY"
    oxide_col: Optional[str] = "OXIDE"
    xinc_col: Optional[str] = None
    yinc_col: Optional[str] = None
    zinc_col: Optional[str] = None
    default_density: float = 2.7

    block_x: Optional[float] = None
    block_y: Optional[float] = None
    block_z: Optional[float] = None

    origin_x: Optional[float] = None
    origin_y: Optional[float] = None
    origin_z: Optional[float] = None

    cutoff: float = 0.5
    smu_x: float = 10.0
    smu_y: Optional[float] = None
    smu_z: Optional[float] = None
    step_x: float = 1.0

    step_y: Optional[float] = None
    mode: str = "MII"
    skip_rows: int = 0
    threads: int = 1

    extra_numeric: list = field(default_factory=list)
    extra_categorical: list = field(default_factory=list)
    extra_metal: list = field(default_factory=list)

    ore_rule: str = "recover"

    elos: float = 0.0

    elos_axes: str = "xy"

    regions: list = field(default_factory=list)
    outside: str = "model"

    sample_size: Optional[float] = None

    audit_fields: bool = False

    dilute_attr_col: Optional[str] = None
    dilute_attr_values: list = field(default_factory=list)

    merge_subcells: bool = False

    merge_rules: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.mode not in ("MI", "MII"):
            raise ValueError("Dilution run must be 'MI' or 'MII'")
        for name in ("smu_x", "step_x"):
            v = getattr(self, name)
            if v is None or v <= 0:
                raise ValueError(f"{name} must be positive")
        if self.step_y is not None and self.step_y <= 0:
            raise ValueError("step_y must be positive")
        m = self.smu_x / self.step_x
        if abs(m - round(m)) > _TOL:
            raise ValueError(f"SMU X ({self.smu_x}) must be a whole number of "
                             f"X steps ({self.step_x})")
        if self.ore_rule not in ORE_RULES:
            raise ValueError(f"Ore rule must be one of {', '.join(ORE_RULES)}")
        if self.outside not in OUTSIDE:
            raise ValueError(f"Outside regions must be one of {', '.join(OUTSIDE)}")
        if self.elos_axes not in ELOS_AXES:
            raise ValueError(f"Edge dilution axes must be one of {', '.join(ELOS_AXES)}")
        e = self.elos or 0.0
        if e < 0:
            raise ValueError("Edge dilution (ELOS) can't be negative")
        if abs(e / self.step_x - round(e / self.step_x)) > _TOL:
            raise ValueError(f"Edge dilution ({e} m) must be a whole number of "
                             f"X steps ({self.step_x} m)")
        if (self.elos_axes == "xy" and self.step_y
                and abs(e / self.step_y - round(e / self.step_y)) > _TOL):
            raise ValueError(f"Edge dilution ({e} m) must be a whole number of "
                             f"Y steps ({self.step_y} m)")
        if self.sample_size is not None and self.sample_size <= 0:
            raise ValueError("The region piece size must be positive")
        bad = [v for v in (self.merge_rules or {}).values() if v not in MERGE_RULES]
        if bad:
            raise ValueError(f"Merge rule must be one of {', '.join(MERGE_RULES)}")
        if self.regions:
            order_regions(as_regions(self.regions))

def common_cell(smu: float, parent: float, max_split: int = 100) -> float:
    r = Fraction(smu / parent).limit_denominator(max_split)
    if r <= 0 or abs(float(r) - smu / parent) > _TOL:
        return 0.0
    return parent / r.denominator

def check_smu_ratio(axis: str, smu: float, parent: float) -> float:
    g = common_cell(smu, parent)
    if g <= 0:
        raise ValueError(
            f"SMU {axis} ({smu} m) and the parent cell {axis} ({parent} m) have "
            f"no common sub-cell size of at least {parent / 100:g} m, so the "
            "SMU can't be sub-celled on the parent boundaries")
    return g

def load_block_model(p: DilutionParams) -> pd.DataFrame:
    _no_dmx(p.input_path)
    if _ext(p.input_path) == ".dm":
        from .io_dm import read_dm
        df = read_dm(p.input_path)
    else:
        df = pd.read_csv(p.input_path, skiprows=p.skip_rows)
    df.columns = [str(c).strip() for c in df.columns]
    log.info("Loaded %d blocks from %s", len(df), p.input_path)
    return df

def _check_columns(columns, p: DilutionParams) -> None:
    columns = set(columns.columns if isinstance(columns, pd.DataFrame) else columns)
    needed = [p.x_col, p.y_col, p.z_col, p.grade_col, p.rescat_col]
    needed += list(p.extra_numeric) + list(p.extra_categorical) + list(p.extra_metal)
    for opt in (p.density_col, p.oxide_col, p.xinc_col, p.yinc_col, p.zinc_col,
                p.dilute_attr_col):
        if opt:
            needed.append(opt)
    missing = [c for c in needed if c not in columns]
    if missing:
        raise ValueError(f"Columns not found in block model: {missing}")

def _min_spacing(values: np.ndarray) -> float:
    u = np.unique(np.round(values, 6))
    if len(u) < 2:
        return float("nan")
    d = np.diff(u)
    d = d[d > 1e-6]
    return float(d.min()) if len(d) else float("nan")

def infer_block_size(src, p: DilutionParams) -> tuple[float, float, float]:
    store = as_store(src, p)
    out = []
    for given, inc, coord in ((p.block_x, p.xinc_col, p.x_col),
                              (p.block_y, p.yinc_col, p.y_col),
                              (p.block_z, p.zinc_col, p.z_col)):
        if given:
            out.append(float(given))
        elif inc:
            vals = store.num(inc)
            mx = [np.nanmax(vals[sl]) for sl in store.chunks()
                  if np.isfinite(vals[sl]).any()]
            out.append(float(max(mx)) if mx else float("nan"))
        else:
            vals = store.num(coord)
            u = np.zeros(0)
            for sl in store.chunks():
                u = np.union1d(u, np.unique(np.round(vals[sl], 6)))
            out.append(_min_spacing(u))
    if any(not np.isfinite(v) or v <= 0 for v in out):
        raise ValueError("Could not infer parent cell size; enter it manually")
    return out[0], out[1], out[2]

def infer_origin(lo: np.ndarray, size: np.ndarray, parent: float,
                 given: Optional[float], axis: str) -> float:
    lo_min = float(lo.min())
    full = np.abs(size - parent) < _TOL
    vals, counts = np.unique(np.mod(lo[full] - lo_min, parent), return_counts=True)
    return _origin(lo_min, vals, counts, parent, given, axis)

def _origin(lo_min: float, vals: np.ndarray, counts: np.ndarray, parent: float,
            given: Optional[float], axis: str) -> float:
    if given is not None:
        if lo_min < given - _TOL:
            raise ValueError(f"Model origin {axis} ({given}) is above the lowest "
                             f"block edge ({lo_min})")
        return float(given)
    total = int(counts.sum())
    if total:
        cum = np.cumsum(counts)
        a = vals[np.searchsorted(cum, (total - 1) // 2, side="right")]
        b = vals[np.searchsorted(cum, total // 2, side="right")]
        ref = float((a + b) / 2) + lo_min
    else:
        log.warning("No whole parent cells found on %s; parent grid anchored "
                    "on the lowest block edge. Enter the model origin if the "
                    "diluted model does not line up.", axis)
        ref = lo_min
    return ref - math.ceil((ref - lo_min) / parent - _EPS) * parent

def block_extents(store: BlockStore, p: DilutionParams, parent, sl) -> dict:
    bx, by, bz = parent
    X = np.asarray(store.num(p.x_col)[sl], dtype=float)
    k = len(X)
    return dict(
        X=X, Y=np.asarray(store.num(p.y_col)[sl], dtype=float),
        Z=np.asarray(store.num(p.z_col)[sl], dtype=float),
        xs=np.asarray(store.num(p.xinc_col)[sl], float) if p.xinc_col else np.full(k, bx),
        ys=np.asarray(store.num(p.yinc_col)[sl], float) if p.yinc_col else np.full(k, by),
        zs=np.asarray(store.num(p.zinc_col)[sl], float) if p.zinc_col else np.full(k, bz))

def model_grid(store: BlockStore, p: DilutionParams, parent=None) -> dict:
    bx, by, bz = parent or infer_block_size(store, p)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for sl in store.chunks():
        e = block_extents(store, p, (bx, by, bz), sl)
        if not len(e["X"]):
            continue
        for a, (c, w) in enumerate(((e["X"], e["xs"]), (e["Y"], e["ys"]),
                                    (e["Z"], e["zs"]))):
            lo[a] = min(lo[a], float((c - w / 2).min()))
            hi[a] = max(hi[a], float((c + w / 2).max()))
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("The model has no blocks, or blank or non-numeric "
                         "coordinates or cell sizes")
    bad = 0
    for sl in store.chunks():
        e = block_extents(store, p, (bx, by, bz), sl)
        bad += int((~np.isfinite(np.stack([e["X"], e["Y"], e["Z"], e["xs"],
                                           e["ys"], e["zs"]]))).any(axis=0).sum())
    if bad:
        raise ValueError(f"{bad:,} blocks have blank or non-numeric coordinates "
                         "or cell sizes")
    sizes = (bx, by, bz)
    given = (p.origin_x, p.origin_y, p.origin_z)
    counts = [dict() for _ in range(3)]
    if any(g is None for g in given):
        for sl in store.chunks():
            e = block_extents(store, p, sizes, sl)
            for a, (c, w) in enumerate(((e["X"], e["xs"]), (e["Y"], e["ys"]),
                                        (e["Z"], e["zs"]))):
                if given[a] is not None:
                    continue
                full = np.abs(w - sizes[a]) < _TOL
                v, k = np.unique(np.mod((c - w / 2)[full] - lo[a], sizes[a]),
                                 return_counts=True)
                d = counts[a]
                for vi, ki in zip(v.tolist(), k.tolist()):
                    d[vi] = d.get(vi, 0) + ki
    origin = []
    for a, axis in enumerate("XYZ"):
        v = np.array(sorted(counts[a]), dtype=float)
        k = np.array([counts[a][x] for x in sorted(counts[a])], dtype=np.int64)
        origin.append(_origin(float(lo[a]), v, k, sizes[a], given[a], axis))
    x0, y0, z0 = origin
    return dict(
        bx=bx, by=by, bz=bz, x0=x0, y0=y0, z0=z0,
        xlo=float(lo[0]), ylo=float(lo[1]), zlo=float(lo[2]),
        xhi=float(hi[0]), yhi=float(hi[1]), zhi=float(hi[2]),
        npx=int(math.ceil((float(hi[0]) - x0) / bx - _EPS)),
        npy=int(math.ceil((float(hi[1]) - y0) / by - _EPS)),
        npz=int(math.ceil((float(hi[2]) - z0) / bz - _EPS)))

def rows_in_z(store: BlockStore, p: DilutionParams, parent, lo: float,
              hi: float) -> np.ndarray:
    def z_range(sl):
        e = block_extents(store, p, parent, sl)
        return e["Z"] - e["zs"] / 2, e["Z"] + e["zs"] / 2
    zmin, order, tall = store.z_index((p.zinc_col, parent[2]), z_range)
    i0 = np.searchsorted(zmin, lo - tall - _TOL, side="left")
    i1 = np.searchsorted(zmin, hi, side="left")
    rows = np.sort(order[i0:i1])
    if not len(rows):
        return rows
    e = block_extents(store, p, parent, rows)
    return rows[e["Z"] + e["zs"] / 2 > lo]

def _explode(lo: np.ndarray, hi: np.ndarray, origin: float, step: float):
    i0 = np.floor((lo - origin) / step + _EPS).astype(np.int64)
    i1 = np.ceil((hi - origin) / step - _EPS).astype(np.int64)
    i1 = np.maximum(i1, i0 + 1)
    n = i1 - i0
    rep = np.repeat(np.arange(len(lo)), n)
    offs = np.repeat(np.cumsum(n) - n, n)
    idx = i0[rep] + (np.arange(len(rep)) - offs)
    a = np.maximum(lo[rep], origin + idx * step)
    b = np.minimum(hi[rep], origin + (idx + 1) * step)
    frac = np.clip(b - a, 0, None) / (hi - lo)[rep]
    return rep, idx, frac

def optimise_segments(values: list[np.ndarray], m: int,
                      relax_start, relax_end,
                      captured: Optional[list[np.ndarray]] = None) -> list[np.ndarray]:
    if captured is not None:
        return _optimise_recover(values, captured, m, relax_start, relax_end)
    if isinstance(relax_start, bool):
        relax_start = [relax_start] * len(values)
    if isinstance(relax_end, bool):
        relax_end = [relax_end] * len(values)
    out: list[Optional[np.ndarray]] = [None] * len(values)
    todo = []
    for i, v in enumerate(values):
        if len(v) < m or not (v >= 0).any():
            out[i] = np.zeros(len(v), dtype=np.int8)
        else:
            todo.append(i)
    todo.sort(key=lambda i: len(values[i]))
    maxL = 2 * m - 1
    budget = 4_000_000
    pos = 0
    while pos < len(todo):
        end = pos
        while end < len(todo):
            lmax = len(values[todo[end]])
            if (end - pos + 1) * (lmax + 1) > budget and end > pos:
                break
            end += 1
        batch = todo[pos:end]
        pos = end
        lmax = max(len(values[i]) for i in batch)
        B = len(batch)
        n_arr = np.array([len(values[i]) for i in batch])
        rs = np.array([bool(relax_start[i]) for i in batch])
        re_ = np.array([bool(relax_end[i]) for i in batch])
        P = np.zeros((B, lmax + 1))
        for r, i in enumerate(batch):
            c = np.cumsum(values[i])
            P[r, 1:len(c) + 1] = c
            P[r, len(c) + 1:] = c[-1]
        Ov = np.full((B, lmax + 1), -np.inf)
        Wv = np.full((B, lmax + 1), -np.inf)
        best = np.full((B, lmax + 1), -np.inf)
        Ov[:, 0] = Wv[:, 0] = best[:, 0] = 0.0
        chO = np.zeros((B, lmax + 1), dtype=np.int32)
        chW = np.zeros((B, lmax + 1), dtype=np.int32)
        for i in range(1, lmax + 1):
            at_end = (n_arr == i) & re_
            for L in range(1, min(maxL, i) + 1):
                j = i - L
                base = best[:, j]
                if L >= m:
                    cand = base + (P[:, i] - P[:, j])
                    better = cand > Ov[:, i]
                    Ov[better, i] = cand[better]
                    chO[better, i] = L
                    cand = base
                else:
                    ok = at_end | rs if j == 0 else at_end
                    if not ok.any():
                        continue
                    cand = np.where(ok, base, -np.inf)
                better = cand > Wv[:, i]
                Wv[better, i] = cand[better]
                chW[better, i] = L

            best[:, i] = np.where(Ov[:, i] > Wv[:, i] + 1e-9, Ov[:, i], Wv[:, i])
        for r, idx in enumerate(batch):
            n = n_arr[r]
            lab = np.zeros(n, dtype=np.int8)
            if not np.isfinite(best[r, n]):
                out[idx] = lab
                continue
            i = n
            while i > 0:
                if Ov[r, i] > Wv[r, i] + 1e-9:
                    L = chO[r, i]
                    lab[i - L:i] = 1
                else:
                    L = chW[r, i]
                if L <= 0:
                    raise RuntimeError("Dilution optimiser failed to backtrack")
                i -= L
            out[idx] = lab
    return out

def _run_batches(lengths: list[int], budget: int = 4_000_000):
    todo = sorted(range(len(lengths)), key=lambda i: lengths[i])
    pos = 0
    while pos < len(todo):
        end = pos
        while end < len(todo):
            if (end - pos + 1) * (lengths[todo[end]] + 1) > budget and end > pos:
                break
            end += 1
        yield todo[pos:end]
        pos = end

def _lex_better(a1, a2, b1, b2, t1, t2):
    return (a1 > b1 + t1) | ((a1 >= b1 - t1) & (a2 > b2 + t2))

def _optimise_recover(values, captured, m, relax_start, relax_end):
    n_runs = len(values)
    if isinstance(relax_start, bool):
        relax_start = [relax_start] * n_runs
    if isinstance(relax_end, bool):
        relax_end = [relax_end] * n_runs
    out: list[Optional[np.ndarray]] = [None] * n_runs
    todo = []
    for i, v in enumerate(values):
        if len(v) < m or not (v >= 0).any():
            out[i] = np.zeros(len(v), dtype=np.int8)
        else:
            todo.append(i)
    maxL = 2 * m - 1
    ninf = -np.inf
    for sub in _run_batches([len(values[i]) for i in todo]):
        batch = [todo[i] for i in sub]
        lmax = max(len(values[i]) for i in batch)
        B = len(batch)
        n_arr = np.array([len(values[i]) for i in batch])
        rs = np.array([bool(relax_start[i]) for i in batch])
        re_ = np.array([bool(relax_end[i]) for i in batch])
        P = np.zeros((B, lmax + 1))
        Q = np.zeros((B, lmax + 1))
        for r, i in enumerate(batch):
            c, q = np.cumsum(values[i]), np.cumsum(captured[i])
            P[r, 1:len(c) + 1], P[r, len(c) + 1:] = c, c[-1]
            Q[r, 1:len(q) + 1], Q[r, len(q) + 1:] = q, q[-1]

        t2 = 1e-9 * np.array([1.0 + np.abs(values[i]).sum() for i in batch])
        t1 = 1e-9 * np.array([1.0 + np.abs(captured[i]).sum() for i in batch])
        O1 = np.full((B, lmax + 1), ninf)
        O2 = np.full((B, lmax + 1), ninf)
        W1 = np.full((B, lmax + 1), ninf)
        W2 = np.full((B, lmax + 1), ninf)
        b1 = np.full((B, lmax + 1), ninf)
        b2 = np.full((B, lmax + 1), ninf)
        O1[:, 0] = O2[:, 0] = W1[:, 0] = W2[:, 0] = b1[:, 0] = b2[:, 0] = 0.0
        chO = np.zeros((B, lmax + 1), dtype=np.int32)
        chW = np.zeros((B, lmax + 1), dtype=np.int32)
        for i in range(1, lmax + 1):
            at_end = (n_arr == i) & re_
            for L in range(1, min(maxL, i) + 1):
                j = i - L
                c1, c2 = b1[:, j], b2[:, j]
                if L >= m:
                    seg = P[:, i] - P[:, j]
                    ok = seg >= -t2
                    o1 = np.where(ok, c1 + (Q[:, i] - Q[:, j]), ninf)
                    o2 = np.where(ok, c2 + seg, ninf)
                    better = _lex_better(o1, o2, O1[:, i], O2[:, i], t1, t2)
                    O1[better, i], O2[better, i] = o1[better], o2[better]
                    chO[better, i] = L
                else:
                    ok = at_end | rs if j == 0 else at_end
                    if not ok.any():
                        continue
                    c1, c2 = np.where(ok, c1, ninf), np.where(ok, c2, ninf)
                better = _lex_better(c1, c2, W1[:, i], W2[:, i], t1, t2)
                W1[better, i], W2[better, i] = c1[better], c2[better]
                chW[better, i] = L

            ore_i = _lex_better(O1[:, i], O2[:, i], W1[:, i], W2[:, i], t1, t2)
            b1[:, i] = np.where(ore_i, O1[:, i], W1[:, i])
            b2[:, i] = np.where(ore_i, O2[:, i], W2[:, i])
        for r, idx in enumerate(batch):
            n = n_arr[r]
            lab = np.zeros(n, dtype=np.int8)
            if not np.isfinite(b1[r, n]):
                out[idx] = lab
                continue
            i = n
            while i > 0:
                if _lex_better(O1[r, i], O2[r, i], W1[r, i], W2[r, i], t1[r], t2[r]):
                    L = chO[r, i]
                    lab[i - L:i] = 1
                else:
                    L = chW[r, i]
                if L <= 0:
                    raise RuntimeError("Dilution optimiser failed to backtrack")
                i -= L
            out[idx] = lab
    return out

def breakdown_feasible(P: np.ndarray, zs: int, ze: int, m: int) -> Optional[list[tuple[int, int]]]:
    def ok(a, b):
        return P[b] - P[a] >= -1e-9 * (1.0 + abs(P[b]) + abs(P[a]))

    blocks, c = [], zs
    for w in breakdown_zone(ze - zs, m):
        blocks.append((c, c + w))
        c += w
    if all(ok(a, b) for a, b in blocks):
        return blocks

    feasible = np.zeros(ze - zs + 1, dtype=bool)
    feasible[ze - zs] = True
    for i in range(ze - m, zs - 1, -1):
        feasible[i - zs] = any(feasible[i + L - zs] and ok(i, i + L)
                               for L in range(m, 2 * m) if i + L <= ze)
    if not feasible[0]:
        return None
    blocks, i = [], zs
    while i < ze:
        L = next(L for L in range(m, 2 * m)
                 if i + L <= ze and feasible[i + L - zs] and ok(i, i + L))
        blocks.append((i, i + L))
        i += L
    return blocks

def breakdown_zone(width: int, m: int) -> list[int]:
    nb = max(1, width // m)
    return [m] * (nb - 1) + [width - (nb - 1) * m]

def refine_zone(P: np.ndarray, zs: int, ze: int, m: int,
                left_free: bool, right_free: bool) -> list[tuple[int, int]]:
    def V(a, b):
        return P[b] - P[a]

    blocks, c = [], zs
    for w in breakdown_zone(ze - zs, m):
        blocks.append((c, c + w))
        c += w

    pieces: list[list[int]] = []
    for a, b in blocks:
        kind = 1 if V(a, b) >= -1e-9 else 0
        if pieces and kind == 0 and pieces[-1][0] == 0:
            pieces[-1][2] = b
        else:
            pieces.append([kind, a, b])
    if all(k == 1 for k, _, _ in pieces):
        return [(a, b) for _, a, b in pieces]

    cap = 2 * m - 1
    i = 0
    while i < len(pieces):
        kind, ws, we = pieces[i]
        if kind == 1:
            i += 1
            continue
        prev = pieces[i - 1] if i > 0 else None
        nxt = pieces[i + 1] if i + 1 < len(pieces) else None
        free = (prev is None and left_free) or (nxt is None and right_free)
        a_max = cap - (prev[2] - prev[1]) if prev else 0
        d_lo = -(cap - (nxt[2] - nxt[1])) if nxt else 0
        d_hi = (nxt[2] - nxt[1]) - m if nxt else 0
        best = (V(prev[1], prev[2]) if prev else 0.0) + \
            (V(nxt[1], nxt[2]) if nxt else 0.0)
        best_ad = (0, 0)
        for a in range(0, max(a_max, 0) + 1):
            for d in range(d_lo, d_hi + 1):
                nws, nwe = ws + a, we + d
                width = nwe - nws
                if width < 0 or (width < m and not free):
                    continue
                val = 0.0
                if prev:
                    pv = V(prev[1], nws)
                    if pv < -1e-9:
                        continue
                    val += pv
                if nxt:
                    nv = V(nwe, nxt[2])
                    if nv < -1e-9:
                        continue
                    val += nv
                if val > best + 1e-9 or (abs(val - best) <= 1e-9 and
                                         abs(a) + abs(d) < sum(map(abs, best_ad))):
                    best, best_ad = val, (a, d)
        a, d = best_ad
        if prev:
            prev[2] = ws + a
        if nxt:
            nxt[1] = we + d
        pieces[i][1], pieces[i][2] = ws + a, we + d
        i += 1
    return [(a, b) for k, a, b in pieces if k == 1 and b > a]

class DilutionCancelled(Exception):
    pass

def _encode(f: pd.DataFrame, sort: bool, grid, names: dict,
            layout: Optional[tuple] = None) -> Optional[dict]:
    if not len(f):
        return None
    if sort:
        f = f.sort_values(["ZC", "YC", "X_FROM"], ascending=[False, True, True],
                          kind="stable")
    if layout is not None:
        cols, grid_cols = layout
        if grid is not None and grid_cols:
            f = _add_parent_grid(f, *grid)
        out = pd.DataFrame({name: f[key].to_numpy() for key, name in cols + grid_cols})
        return dict(ids=False, columns=list(out.columns), zc=f["ZC"].to_numpy(float),
                    body=out.to_csv(header=False, index=False))
    f = f[[c for c in f.columns if not c.startswith(OWN)]]
    if grid is not None:
        f = _add_parent_grid(f, *grid)
    f = f.rename(columns=names)
    if list(f.columns[:2]) != ["BLOCK_ID", "ZONE_ID"]:
        raise RuntimeError("BLOCK_ID and ZONE_ID must be the first columns")
    return dict(ids=True, columns=list(f.columns), bid=f["BLOCK_ID"].to_numpy(np.int64),
                zid=f["ZONE_ID"].to_numpy(np.int64), zc=f["ZC"].to_numpy(float),
                body=f.iloc[:, 2:].to_csv(header=False, index=False))

def _decode(enc: dict, bid_off: int, zone_off: int) -> list[bytes]:
    lines = enc["body"].split(os.linesep)
    sep = os.linesep
    if not enc.get("ids", True):
        return [f"{t}{sep}".encode("utf-8") for t in lines[:len(enc["zc"])]]
    ore = enc["zid"] > 0
    bid = np.where(ore, enc["bid"] + bid_off, enc["bid"]).tolist()
    zid = np.where(ore, enc["zid"] + zone_off, enc["zid"]).tolist()
    return [f"{b},{z},{t}{sep}".encode("utf-8") for b, z, t in zip(bid, zid, lines)]

GRID_FIELDS = ("IJK", "XMORIG", "YMORIG", "ZMORIG", "NX", "NY", "NZ")

def dm_layout(p: DilutionParams, extras: dict, header) -> tuple[list, list]:
    cols = [("MATERIAL", "MATERIAL"), ("XC", p.x_col), ("YC", p.y_col),
            ("ZC", p.z_col), ("XINC", p.xinc_col or "XINC"),
            ("YINC", p.yinc_col or "YINC"), ("ZINC", p.zinc_col or "ZINC")]
    used = {name.upper() for _, name in cols}

    def add(key, name):
        if name.upper() not in used:
            cols.append((key, name))
            used.add(name.upper())

    add(OWN + "AU", p.grade_col)
    add("AU", f"D_{p.grade_col}")
    add(OWN + "RESCAT", p.rescat_col)
    add("RESCAT", f"D_{p.rescat_col}")
    if p.density_col:
        add("DENSITY", p.density_col)
    if p.oxide_col:
        add(OWN + "OXIDE", p.oxide_col)
        add("OXIDE", f"D_{p.oxide_col}")
    for src, out_name in extras["numeric"]:
        add(OWN + out_name, src)
        add(out_name, f"D_{out_name}")
    for src, out_name, _, _ in extras["categorical"]:
        add(OWN + out_name, src)
        add(out_name, f"D_{out_name}")
    upper = {str(h).strip().upper(): str(h).strip() for h in header}
    grid_cols = [(g, upper[g]) for g in GRID_FIELDS if g in upper and g not in used]
    return cols, grid_cols

def merge_attributes(p: DilutionParams, extras: dict, layout) -> dict:
    if layout is None:
        return {}
    num = {out for _, out in extras["numeric"]}
    out = {}
    for key, name in layout[0]:
        if key.startswith(OWN):
            out[name] = "num" if key == OWN + "AU" or key[len(OWN):] in num else "cat"
    return out

def _zc_groups(zc: np.ndarray):
    cuts = [0, *(np.flatnonzero(np.diff(zc)) + 1).tolist(), len(zc)]
    return list(zip(cuts[:-1], cuts[1:]))

class DilutedSpool:

    def __init__(self, work_dir: str):
        self.dir = work_dir
        self.columns: Optional[list] = None
        self.files: dict = {}
        self.rows = 0
        self._finalizer = weakref.finalize(self, shutil.rmtree, work_dir, True)

    def add(self, enc: Optional[dict], bid_off: int, zone_off: int) -> None:
        if enc is None:
            return
        if self.columns is None:
            self.columns = enc["columns"]
        elif enc["columns"] != self.columns:
            raise RuntimeError("Diluted model columns differ between jobs")
        lines = _decode(enc, bid_off, zone_off)
        for a, b in _zc_groups(enc["zc"]):
            key = float(enc["zc"][a])
            path = self.files.get(key)
            if path is None:
                path = self.files[key] = os.path.join(self.dir, f"dm_{len(self.files):06d}.csv")
            with open(path, "ab") as f:
                f.write(b"".join(lines[a:b]))
        self.rows += len(enc["zc"])

    def add_frame(self, f: pd.DataFrame, zc: str) -> None:
        if not len(f):
            return
        if self.columns is None:
            self.columns = list(f.columns)
        f = f[self.columns]
        z = f[zc].to_numpy(float)
        for a, b in _zc_groups(z):
            key = float(z[a])
            path = self.files.get(key)
            if path is None:
                path = self.files[key] = os.path.join(self.dir, f"dm_{len(self.files):06d}.csv")
            with open(path, "ab") as out:
                out.write(f.iloc[a:b].to_csv(header=False, index=False).encode("utf-8"))
        self.rows += len(f)

    def paths(self) -> list[str]:
        return [self.files[k] for k in sorted(self.files, reverse=True)]

    def write(self, path: str, empty_columns: list) -> None:
        cols = self.columns or empty_columns
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        with open(path, "ab") as out:
            for part in self.paths():
                with open(part, "rb") as f:
                    shutil.copyfileobj(f, out, 16 * 1024 * 1024)

    def levels(self, ascending: bool = False):
        paths = self.paths()
        for part in reversed(paths) if ascending else paths:
            yield pd.read_csv(part, header=None, names=self.columns)

    def chunks(self, rows: int = CHUNK_ROWS):
        for part in self.paths():
            yield from pd.read_csv(part, header=None, names=self.columns,
                                   chunksize=rows)

    def frame(self, empty_columns: list) -> pd.DataFrame:
        parts = [pd.read_csv(f, header=None, names=self.columns) for f in self.paths()]
        if not parts:
            return pd.DataFrame(columns=self.columns or empty_columns)
        return pd.concat(parts, ignore_index=True)

    def drop_model(self) -> None:
        for path in self.files.values():
            try:
                os.remove(path)
            except OSError as e:
                log.debug("Could not delete %s: %s", path, e)
        self.files = {}

    def close(self) -> None:
        self._finalizer()

class OreSpool:

    def __init__(self, work_dir: str, empty_columns: list):
        self.path = os.path.join(work_dir, "ore_blocks.csv")
        open(self.path, "wb").close()
        self.empty_columns = empty_columns
        self.columns: Optional[list] = None
        self.benches: dict = {}
        self.counts: dict = {}
        self.oxides: set = set()
        self.rows = 0

    def add(self, enc: Optional[dict], bid_off: int, zone_off: int,
            oxides=()) -> None:
        if enc is None:
            return
        if self.columns is None:
            self.columns = enc["columns"]
        elif enc["columns"] != self.columns:
            raise RuntimeError("Ore block columns differ between jobs")
        lines = _decode(enc, bid_off, zone_off)
        with open(self.path, "ab") as f:
            for a, b in _zc_groups(enc["zc"]):
                start = f.tell()
                f.write(b"".join(lines[a:b]))
                key = float(enc["zc"][a])
                self.benches.setdefault(key, []).append((start, f.tell()))
                self.counts[key] = self.counts.get(key, 0) + (b - a)
        self.oxides.update(oxides)
        self.rows += len(enc["zc"])

    def _read(self, source) -> pd.DataFrame:
        return pd.read_csv(source, header=None, names=self.columns)

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=self.empty_columns)
        return self._read(self.path)

    def chunks(self, rows: int = CHUNK_ROWS):
        if self.rows:
            yield from pd.read_csv(self.path, header=None, names=self.columns,
                                   chunksize=rows)

    def bench(self, zc: float) -> pd.DataFrame:
        key = next((k for k in self.benches if abs(k - zc) < 1e-6), None)
        if key is None:
            return pd.DataFrame(columns=self.columns or self.empty_columns)
        import io
        with open(self.path, "rb") as f:
            data = b""
            for a, b in self.benches[key]:
                f.seek(a)
                data += f.read(b - a)
        return self._read(io.BytesIO(data))

    def write(self, path: str) -> None:
        pd.DataFrame(columns=self.columns or self.empty_columns).to_csv(path, index=False)
        with open(path, "ab") as out, open(self.path, "rb") as f:
            shutil.copyfileobj(f, out, 16 * 1024 * 1024)

@dataclass
class DilutionResult:
    summary: pd.DataFrame
    by_bench: pd.DataFrame
    params: dict
    stats: dict = field(default_factory=dict)
    ore_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spool: Optional[DilutedSpool] = None
    store: Optional[BlockStore] = None
    grid: dict = field(default_factory=dict)
    empty_columns: list = field(default_factory=list)
    run_params: Optional[DilutionParams] = None
    ore_spool: Optional[OreSpool] = None
    layout: Optional[tuple] = None
    by_region: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def ore_blocks(self) -> pd.DataFrame:
        return self.ore_spool.frame()

    def ore_chunks(self, rows: int = CHUNK_ROWS):
        return self.ore_spool.chunks(rows)

    def close(self) -> None:
        if self.spool is not None:
            self.spool.close()
        if self.store is not None and self.store.work_dir:
            self.store.close()

    def bench_cells(self, zc: float) -> dict:
        p, pr = self.run_params, self.params
        parent = (pr["block_x"], pr["block_y"], pr["block_z"])
        sz = pr["smu_z"]
        rows = rows_in_z(self.store, p, parent, zc - sz / 2 + 1e-6, zc + sz / 2 - 1e-6)
        e = block_extents(self.store, p, parent, rows)
        keep = e["Z"] - e["zs"] / 2 < zc + sz / 2 - 1e-6
        if p.density_col:
            d = np.asarray(self.store.num(p.density_col)[rows], float)
            keep &= ~(np.isfinite(d) & (d <= 0))
        rows = rows[keep]
        e = {k: v[keep] for k, v in e.items()}
        e["def"] = definition_grade(np.asarray(self.store.num(p.grade_col)[rows], float),
                                    np.asarray(self.store.num(p.rescat_col)[rows], float),
                                    p.mode)
        e["ore"] = np.clip(self.ore_frac[rows], 0, 1)
        return e

    @property
    def diluted_model(self) -> pd.DataFrame:
        if self.spool is None:
            return pd.DataFrame(columns=self.empty_columns)
        return self.spool.frame(self.empty_columns)

    def flagged_chunks(self, rows: int = CHUNK_ROWS):
        p = self.params
        src = self.store.source if self.store is not None else p["input_path"]
        if isinstance(src, pd.DataFrame):
            chunks = (src.iloc[s:s + rows] for s in range(0, max(len(src), 1), rows))
        else:
            chunks = iter_chunks(p["input_path"], p.get("skip_rows", 0), rows=rows)
        start = 0
        for ch in chunks:
            ch = ch.copy()
            k = len(ch)
            ch["ORE_FRAC"] = self.ore_frac[start:start + k].clip(0, 1)
            ch[def_col(p)] = definition_grade(
                ch[p["grade_col"]], ch[p["rescat_col"]], p["mode"])
            start += k
            yield ch

    @property
    def block_model(self) -> pd.DataFrame:
        return pd.concat(list(self.flagged_chunks()), ignore_index=True)

def clean_grade(values) -> np.ndarray:
    g = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float) \
        if not isinstance(values, np.ndarray) else values.astype(float)
    return np.where(~np.isfinite(g) | (g < 0), 0.0, g)

def clean_rescat(values) -> np.ndarray:
    r = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float) \
        if not isinstance(values, np.ndarray) else values.astype(float)
    return np.where(np.isnan(r), 4.0, r)

def definition_grade(grade, rescat, mode: str) -> np.ndarray:
    g, r = clean_grade(grade), clean_rescat(rescat)
    zero = (r == 4) if mode == "MII" else (r >= 3)
    return np.where(zero, 0.0, g)

def _cat_label(c) -> str:
    if isinstance(c, float) and c.is_integer():
        return str(int(c))
    return str(c)

_RESERVED = {"BLOCK_ID", "ZONE_ID", "MATERIAL", "X_FROM", "X_TO", "XC", "XINC",
             "YC", "YINC", "ZC", "ZINC", "TONNES", "VOLUME", "DENSITY", "AU",
             "AU_DEF", "METAL_G", "METAL_OZ", "OXIDE", "OXIDE_PCT",
             "TONNES_ABOVE_COG", "DILUTION_TONNES", "RESCAT", "IJK", "XMORIG", "YMORIG",
             "ZMORIG", "PARENT_XINC", "PARENT_YINC", "PARENT_ZINC", "NX", "NY",
             "NZ"}

def _extra_attributes(store: BlockStore, p: DilutionParams) -> dict:
    def out_name(c):
        return f"{c}_BM" if c.upper() in _RESERVED or c.upper().startswith(
            "TONNES_RESCAT_") else c

    numeric = [(c, out_name(c)) for c in p.extra_numeric]
    categorical = []
    for c, by in ([(c, "mass") for c in p.extra_categorical]
                  + [(c, "metal") for c in p.extra_metal]):
        values = list(store.codes(c)[1]) + ["MISSING"]
        name = out_name(c)
        if by == "metal" and c in p.extra_categorical:
            name = f"{c}_METAL"
        categorical.append((c, name, values, by))
    return {"numeric": numeric, "categorical": categorical}

def _scan_attributes(store: BlockStore, p: DilutionParams) -> dict:
    cats: set = set()
    bad_g = miss_r = void = miss_d = miss_ox = 0
    for sl in store.chunks():
        g = store.num(p.grade_col)[sl]
        bad_g += int((~np.isfinite(g) | (g < 0)).sum())
        r = store.num(p.rescat_col)[sl]
        miss_r += int(np.isnan(r).sum())
        cats.update(np.unique(np.where(np.isnan(r), 4.0, r)).tolist())
        if p.density_col:
            d = store.num(p.density_col)[sl]
            void += int((np.isfinite(d) & (d <= 0)).sum())
            miss_d += int((~np.isfinite(d)).sum())
        if p.oxide_col:
            miss_ox += int((store.codes(p.oxide_col)[0][sl] < 0).sum())
    if bad_g:
        log.warning("%d blocks have missing/negative %s; treated as 0",
                    bad_g, p.grade_col)
    if miss_r:
        log.warning("%d blocks have missing %s; treated as unclassified (4)",
                    miss_r, p.rescat_col)
    if void:
        log.info("%d blocks have zero density and are treated as void/air", void)
    if miss_d:
        log.warning("%d blocks have missing density; using %.3f",
                    miss_d, p.default_density)
    if p.oxide_col:
        ox_values = list(store.codes(p.oxide_col)[1])
        if miss_ox:
            log.warning("%d blocks have missing %s; reported as 'MISSING'",
                        miss_ox, p.oxide_col)
            ox_values.append("MISSING")
    else:
        ox_values = ["NONE"]
    return dict(categories=sorted(cats), ox_values=ox_values)

def _block_fields(store: BlockStore, p: DilutionParams, rows: np.ndarray,
                  parent, categories: list, extras: dict) -> dict:
    e = block_extents(store, p, parent, rows)
    k = len(rows)
    grade = clean_grade(np.asarray(store.num(p.grade_col)[rows], float))
    rescat = clean_rescat(np.asarray(store.num(p.rescat_col)[rows], float))
    if p.density_col:
        dens = np.asarray(store.num(p.density_col)[rows], float)
        void = np.isfinite(dens) & (dens <= 0)
        dens = np.where(~np.isfinite(dens), p.default_density, dens)
    else:
        void = np.zeros(k, dtype=bool)
        dens = np.full(k, p.default_density)
    if p.oxide_col:
        codes, values = store.codes(p.oxide_col)
        ox = np.asarray(codes[rows], dtype=np.int64)
        ox = np.where(ox < 0, len(values), ox)
    else:
        ox = np.zeros(k, dtype=np.int64)
    zero_def = (rescat == 4) if p.mode == "MII" else (rescat >= 3)
    vol = e["xs"] * e["ys"] * e["zs"]
    f = dict(xmin=e["X"] - e["xs"] / 2, xmax=e["X"] + e["xs"] / 2,
             ymin=e["Y"] - e["ys"] / 2, ymax=e["Y"] + e["ys"] / 2,
             zmin=e["Z"] - e["zs"] / 2, zmax=e["Z"] + e["zs"] / 2,
             vol=vol, tonnes=vol * dens, grade=grade,
             def_grade=np.where(zero_def, 0.0, grade),
             cat_idx=np.searchsorted(categories, rescat), ox_codes=ox, void=void)
    for c, _ in extras["numeric"]:
        vals = np.asarray(store.num(c)[rows], float)
        f["xn_" + c] = np.where(vals <= -1e29, np.nan, vals)
    for c, _, values, _ in extras["categorical"]:
        codes = np.asarray(store.codes(c)[0][rows], dtype=np.int64)
        f["xc_" + c] = np.where(codes < 0, len(values) - 1, codes)
    if p.dilute_attr_col:
        ac = p.dilute_attr_col
        a_codes, a_vals = store.codes(ac)
        a = np.asarray(a_codes[rows], dtype=np.int64)
        f["attr_codes"] = np.where(a < 0, len(a_vals), a)
    return f

def def_col(p) -> str:
    g = p["grade_col"] if isinstance(p, dict) else p.grade_col
    mode = p["mode"] if isinstance(p, dict) else p.mode
    return f"{g}_{mode}"

def diluted_model_name(mode: str) -> str:
    return f"diluted_model_{mode}.csv"

def diluted_names(p: DilutionParams, extras: dict) -> dict:
    g = p.grade_col
    ox = p.oxide_col or "OXIDE"
    names = {"AU": f"D_{g}", "AU_DEF": def_col(p), "OXIDE": f"D_{ox}",
             "OXIDE_PCT": f"D_{ox}_PCT", "RESCAT": f"D_{p.rescat_col}"}
    for e in extras["numeric"] + extras["categorical"]:
        names[e[1]] = f"D_{e[1]}"
    return names

def _first_max(v: np.ndarray) -> np.ndarray:
    if not v.shape[1]:
        return np.zeros(len(v), dtype=np.int64)
    mx = v.max(axis=1, keepdims=True)
    return np.argmax(v >= mx - 1e-7 * np.abs(mx), axis=1)

def _dominant(Cmass, Cmetal, a, b):
    mass = Cmass[b] - Cmass[a]
    by_mass = _first_max(mass)
    if Cmetal is None:
        return by_mass
    m_ = Cmetal[b] - Cmetal[a]
    return np.where(m_.max(axis=1) > 1e-9 * mass.max(axis=1), _first_max(m_),
                    by_mass)

def _cum(a: np.ndarray) -> np.ndarray:
    return np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)])

def _new_tot() -> dict:
    return dict(t_in=0.0, m_in=0.0, t_ore=0.0, m_ore=0.0, t_dil=0.0, m_dil=0.0,
                t_lost=0.0, m_lost=0.0, t_elos=0.0, zones=0, recoded=0,
                narrow_waste=0, grown=0, t_grown=0.0, elos_x=0, t_elos_x=0.0,
                elos_y=0, t_elos_y=0.0,

                pd_t_in=0.0, pd_m_in=0.0, pd_t_ore=0.0, pd_m_ore=0.0,
                pd_t_dil=0.0, pd_m_dil=0.0, pd_t_lost=0.0, pd_m_lost=0.0,
                t_oo=0.0, m_oo=0.0, t_ww=0.0, m_ww=0.0,

                rc_pieces=np.zeros(4), rc_cells=np.zeros(4), rc_model=np.zeros(4))

def _source_totals(store: BlockStore, p: DilutionParams, parent) -> np.ndarray:
    out = np.zeros(4)
    for sl in store.chunks():
        e = block_extents(store, p, parent, sl)
        vol = e["xs"] * e["ys"] * e["zs"]
        if p.density_col:
            d = np.asarray(store.num(p.density_col)[sl], float)
            d = np.where(~np.isfinite(d), p.default_density, np.maximum(d, 0.0))
        else:
            d = np.full(len(vol), p.default_density)
        t = vol * d
        g = clean_grade(np.asarray(store.num(p.grade_col)[sl], float))
        out += (len(vol), vol.sum(), t.sum(), (t * g).sum())
    return out

def _log_reconciliation(src: np.ndarray, tot: dict) -> None:
    lines = [f"{'Reconciliation':<16}{'count':>12}{'volume m3':>18}{'tonnes':>18}"
             f"{'metal g':>18}{'d tonnes':>11}{'d metal':>11}"]
    for name, v in (("source", src), ("pieces", tot["rc_pieces"]),
                    ("after dilution", tot["rc_cells"]),
                    ("diluted model", tot["rc_model"])):
        dt = (v[2] / src[2] - 1) if src[2] else 0.0
        dm = (v[3] / src[3] - 1) if src[3] else 0.0
        lines.append(f"{name:<16}{v[0]:12,.0f}{v[1]:18,.3f}{v[2]:18,.3f}{v[3]:18,.3f}"
                     f"{dt:11.1e}{dm:11.1e}")
    log.debug("\n".join(lines))

def _aggregate(A: dict, blk: np.ndarray, frac: np.ndarray, inv: np.ndarray,
               n: int, extras: dict, ncat: int, nox: int):
    tp = A["tonnes"][blk] * frac
    mp = tp * A["grade"][blk]
    g = dict(t=np.bincount(inv, weights=tp, minlength=n),
             v=np.bincount(inv, weights=A["vol"][blk] * frac, minlength=n),
             dm=np.bincount(inv, weights=tp * A["def_grade"][blk], minlength=n),
             rm=np.bincount(inv, weights=mp, minlength=n))
    cat, catm, ox = np.zeros((n, ncat)), np.zeros((n, ncat)), np.zeros((n, nox))
    for ci in range(ncat):
        sel = A["cat_idx"][blk] == ci
        cat[:, ci] = np.bincount(inv[sel], weights=tp[sel], minlength=n)
        catm[:, ci] = np.bincount(inv[sel], weights=mp[sel], minlength=n)
    for oi in range(nox):
        sel = A["ox_codes"][blk] == oi
        ox[:, oi] = np.bincount(inv[sel], weights=tp[sel], minlength=n)
    g.update(cat=cat, catm=catm, ox=ox)
    C = {k: _cum(v) for k, v in g.items()}
    C["num"] = []
    for name, out_name in extras["numeric"]:
        val = A["xn_" + name][blk]
        ok = np.isfinite(val)
        w = np.where(ok, tp, 0.0)
        C["num"].append((out_name,
                         _cum(np.bincount(inv, weights=w * np.where(ok, val, 0.0),
                                          minlength=n)),
                         _cum(np.bincount(inv, weights=w, minlength=n))))
    C["cats"] = []
    for name, out_name, values, by_ in extras["categorical"]:
        code = A["xc_" + name][blk]
        mat = np.zeros((n, len(values)))
        met = np.zeros((n, len(values))) if by_ == "metal" else None
        for vi in range(len(values)):
            sel = code == vi
            mat[:, vi] = np.bincount(inv[sel], weights=tp[sel], minlength=n)
            if met is not None:
                met[:, vi] = np.bincount(inv[sel], weights=mp[sel], minlength=n)
        vv = np.empty(len(values), dtype=object)
        vv[:] = values
        C["cats"].append((out_name, _cum(mat), None if met is None else _cum(met), vv))
    return g, C

OWN = "__own__"
_CARRIED = ["AU", "AU_DEF", "OXIDE", "OXIDE_PCT", "RESCAT"]

def _frame(env: dict, C: dict, xr, a, b, zone, bid, material, geom,
           src=None, own: bool = False, xspan=None) -> pd.DataFrame:
    cat_cols, sy, sz = env["cat_cols"], env["sy"], env["sz"]
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if not len(a):
        return _block_frame([], cat_cols, sy, sz)
    t, v = C["t"][b] - C["t"][a], C["v"][b] - C["v"][a]
    oxt = C["ox"][b] - C["ox"][a]
    dom = _first_max(oxt)
    pos = t > 0
    ts = np.where(pos, t, 1.0)
    yc, yinc, zc_, zinc = geom
    x0, dx = env["x0"], env["dx"]
    cols = {
        "BLOCK_ID": np.asarray(bid, dtype=np.int64),
        "ZONE_ID": np.asarray(zone, dtype=np.int64), "MATERIAL": material,
        "X_FROM": x0 + np.asarray(xr[0]) * dx if xspan is None else xspan[0],
        "X_TO": x0 + np.asarray(xr[1]) * dx if xspan is None else xspan[1],
        "YC": yc, "YINC": yinc, "ZC": zc_, "ZINC": zinc,
        "TONNES": t, "VOLUME": v,
        "DENSITY": np.where(v > 0, t / np.where(v > 0, v, 1.0), 0.0),
        "AU": np.where(pos, (C["rm"][b] - C["rm"][a]) / ts, 0.0),
        "AU_DEF": np.where(pos, (C["dm"][b] - C["dm"][a]) / ts, 0.0),
        "METAL_G": C["rm"][b] - C["rm"][a],
        "OXIDE": env["oxv"][dom],
        "OXIDE_PCT": np.where(pos, 100 * oxt[np.arange(len(a)), dom] / ts, 0.0),
        "TONNES_ABOVE_COG": C["i"][b] - C["i"][a],
        "RESCAT": env["rcv"][_dominant(C["cat"], C["catm"], a, b)],
        "cat": C["cat"][b] - C["cat"][a],
    }
    for out_name, Cn, Cw in C["num"]:
        w = Cw[b] - Cw[a]
        cols[out_name] = np.where(w > 0, (Cn[b] - Cn[a]) / np.where(w > 0, w, 1.0),
                                  np.nan)
    for out_name, Cm, Cmet, vv in C["cats"]:
        cols[out_name] = vv[_dominant(Cm, Cmet, a, b)]
    carried = _CARRIED + [e[0] for e in C["num"]] + [e[0] for e in C["cats"]]
    if own:
        for key in carried:
            cols[OWN + key] = cols[key]
    if src is not None:
        blocks, idx = src
        for key in carried:
            cols[key] = blocks[key].to_numpy()[idx]
        cols["METAL_G"] = t * cols["AU"]
    return _block_frame(cols, cat_cols, sy, sz)

def apply_elos(blocks: list[tuple[int, int]], n: int, m: int,
               e: int) -> list[tuple[int, int]]:
    if not blocks or e <= 0:
        return list(blocks)
    out = [list(b) for b in blocks]
    groups = []
    for i, (s, _) in enumerate(blocks):
        if groups and out[groups[-1][1]][1] == s:
            groups[-1][1] = i
        else:
            groups.append([i, i])
    left = [0] * len(groups)
    right = [0] * len(groups)
    left[0] = min(e, blocks[groups[0][0]][0])
    right[-1] = min(e, n - blocks[groups[-1][1]][1])
    for g in range(len(groups) - 1):
        gap = blocks[groups[g + 1][0]][0] - blocks[groups[g][1]][1]
        spare = max(0, gap - m)
        right[g] = min(e, spare - min(e, spare // 2))
        left[g + 1] = min(e, spare - right[g])
    for (first, last), lo, hi in zip(groups, left, right):
        out[first][0] -= lo
        out[last][1] += hi
    return [(s, t) for s, t in out]

def _dilute_cells(value: np.ndarray, captured: np.ndarray,
                  starts: np.ndarray, ends: np.ndarray, rs: list, re_: list,
                  rp: dict, tot: dict, owned: Optional[np.ndarray] = None) -> dict:
    m, e = rp["m"], rp["elos"]
    nc = len(value)
    ore = np.zeros(nc, dtype=np.int8)
    oa, ob, oz, da, db = [], [], [], [], []
    zones = 0
    zone0 = None
    if owned is None:
        owned = np.ones(len(starts), dtype=bool)
    if rp["no_ore"]:
        labels = [np.zeros(b - a, dtype=np.int8) for a, b in zip(starts, ends)]
    else:
        runs = [value[a:b] for a, b in zip(starts, ends)]
        cap = ([captured[a:b] for a, b in zip(starts, ends)]
               if rp["rule"] == "recover" else None)
        labels = optimise_segments(runs, m, rs, re_, captured=cap)
    for r, (a, b, lab) in enumerate(zip(starts, ends, labels)):
        mine = bool(owned[r])
        if mine and zone0 is None:
            zone0 = zones
        if lab.any():
            P = np.concatenate([[0.0], np.cumsum(value[a:b])])
            d = np.diff(np.concatenate([[0], lab, [0]]))
            design = []
            for zs_, ze_ in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                zones += 1
                tot["zones"] += mine
                blocks = None
                if rp["rule"] == "recover":
                    blocks = breakdown_feasible(P, int(zs_), int(ze_), m)
                if blocks is None:
                    left_free = zs_ > 0 or bool(rs[r])
                    right_free = ze_ < b - a or bool(re_[r])
                    blocks = refine_zone(P, int(zs_), int(ze_), m,
                                         left_free, right_free)
                    nbd = len(breakdown_zone(int(ze_ - zs_), m))
                    tot["recoded"] += max(0, nbd - len(blocks)) * mine
                design += [(s_, e_, zones) for s_, e_ in blocks]
            final = apply_elos([(s_, e_) for s_, e_, _ in design], b - a, m, e)
            for (s_, e_, z), (fs, fe) in zip(design, final):
                ore[a + fs:a + fe] = 1
                oa.append(a + fs), ob.append(a + fe), oz.append(z)
                da.append(a + s_), db.append(a + e_)
        if mine:

            rr = ore[a:b]
            d = np.diff(np.concatenate([[1], rr, [1]]))
            for ws_, we_ in zip(np.flatnonzero(d == -1), np.flatnonzero(d == 1)):
                if ws_ > 0 and we_ < b - a and we_ - ws_ < m:
                    tot["narrow_waste"] += 1
    return dict(ore=ore, oa=np.asarray(oa, dtype=np.int64),
                ob=np.asarray(ob, dtype=np.int64), oz=np.asarray(oz, dtype=np.int64),
                da=np.asarray(da, dtype=np.int64), db=np.asarray(db, dtype=np.int64),
                zone0=zone0 or 0)

def _bench_model(env, A, blk, frac, row, iyq, kx, zp, extras, ncat, nox,
                 ore_blocks, bids, oz, lab_p, ins_p, geo):
    y0, z0, dy, ny, sz, by, bz, bx, dx, iz, yq, nk = geo
    sy = env["sy"]
    if not len(blk):
        return _block_frame([], env["cat_cols"], sy, sz)
    yp = iyq // yq

    lo = [int(v.min()) for v in (row, yp, zp)]
    ny_, nz_ = int(yp.max()) - lo[1] + 1, int(zp.max()) - lo[2] + 1
    ukey, pinv = np.unique(((row - lo[0]) * ny_ + (yp - lo[1])) * nz_ + (zp - lo[2]),
                           return_inverse=True)
    upiece = np.stack([ukey // (ny_ * nz_) + lo[0], (ukey // nz_) % ny_ + lo[1],
                       ukey % nz_ + lo[2]], axis=1)

    b_lo, nbk = int(blk.min()), int(blk.max() - blk.min()) + 1
    ukb, pinv = np.unique(pinv.ravel() * nbk + (blk - b_lo), return_inverse=True)
    pinv = pinv.ravel()
    upiece = np.column_stack([upiece[ukb // nbk], ukb % nbk + b_lo])
    ufine, finv = np.unique(pinv * nk + kx, return_inverse=True)
    finv = finv.ravel()
    nf = len(ufine)
    f_pid, f_k = ufine // nk, ufine % nk
    f_lab = np.zeros(nf, dtype=np.int64)
    f_lab[finv] = lab_p
    f_ins = np.zeros(nf, dtype=bool)
    f_ins[finv] = ins_p
    f_px = np.floor(f_k * dx / bx + _EPS).astype(np.int64)

    new = np.ones(nf, dtype=bool)
    new[1:] = ((f_pid[1:] != f_pid[:-1]) | (f_k[1:] != f_k[:-1] + 1)
               | (f_px[1:] != f_px[:-1]) | (f_lab[1:] != f_lab[:-1]))
    st = np.flatnonzero(new)
    run_of = np.cumsum(new) - 1
    r_pid, r_lab, r_k0 = f_pid[st], f_lab[st], f_k[st]
    r_k1 = np.append(f_k[st[1:] - 1], f_k[-1]) if len(st) else st
    r_row, r_yp, r_zp = upiece[r_pid, 0], upiece[r_pid, 1], upiece[r_pid, 2]
    r_blk = upiece[r_pid, 3]
    r_strip = r_row // ny

    o = np.lexsort((r_row, r_blk, r_k1, r_k0, r_lab, r_zp, r_yp, r_strip))
    same = np.zeros(len(o), dtype=bool)
    if len(o) > 1:
        a_, b_ = o[1:], o[:-1]
        same[1:] = ((r_strip[a_] == r_strip[b_]) & (r_yp[a_] == r_yp[b_])
                    & (r_zp[a_] == r_zp[b_]) & (r_lab[a_] == r_lab[b_])
                    & (r_blk[a_] == r_blk[b_])
                    & (r_k0[a_] == r_k0[b_]) & (r_k1[a_] == r_k1[b_])
                    & (r_row[a_] == r_row[b_] + 1))
    rect_sorted = np.cumsum(~same) - 1
    rect_of_run = np.empty(len(o), dtype=np.int64)
    rect_of_run[o] = rect_sorted

    f_rect = rect_of_run[run_of]
    perm = np.lexsort((f_k, upiece[f_pid, 0], f_rect))
    pos = np.empty(nf, dtype=np.int64)
    pos[perm] = np.arange(nf)
    _, C = _aggregate(A, blk, frac, pos[finv], nf, extras, ncat, nox)
    t_f = np.diff(C["t"])
    C["i"] = np.concatenate([[0.0], np.cumsum(np.where(f_ins[perm], t_f, 0.0))])
    fr = f_rect[perm]
    a = np.flatnonzero(np.r_[True, fr[1:] != fr[:-1]])
    b = np.append(a[1:], nf)
    first = o[np.flatnonzero(~same)]
    last = o[np.append(np.flatnonzero(~same)[1:] - 1, len(o) - 1)]
    k0, k1 = r_k0[first], r_k1[first] + 1
    lab = r_lab[first]
    rowa, rowb = r_row[first], r_row[last] + 1
    ylo = np.maximum(y0 + rowa * dy, y0 + r_yp[first] * by)
    yhi = np.minimum(y0 + rowb * dy, y0 + (r_yp[first] + 1) * by)
    zb_lo = z0 + iz * sz
    zlo = np.maximum(zb_lo, z0 + r_zp[first] * bz)
    zhi = np.minimum(zb_lo + sz, z0 + (r_zp[first] + 1) * bz)
    xlo, xhi = env["x0"] + k0 * dx, env["x0"] + k1 * dx

    bk = r_blk[first]
    for lo_, hi_, key in ((xlo, xhi, "x"), (ylo, yhi, "y"), (zlo, zhi, "z")):
        bmin, bmax = A[key + "min"][bk], A[key + "max"][bk]
        lo_[:] = np.where(bmin > lo_ + _TOL, bmin, lo_)
        hi_[:] = np.where(bmax < hi_ - _TOL, bmax, hi_)
    geom = ((ylo + yhi) / 2, yhi - ylo, (zlo + zhi) / 2, zhi - zlo)
    ore = lab >= 0
    parts = []
    for sel, material in ((np.flatnonzero(ore), "ORE"), (np.flatnonzero(~ore), "WASTE")):
        if not len(sel):
            continue
        li = lab[sel]
        g = tuple(v[sel] for v in geom)
        if material == "ORE":
            parts.append(_frame(env, C, (k0[sel], k1[sel]), a[sel], b[sel], oz[li],
                                bids[li], "ORE", g, (ore_blocks, li), own=True,
                                xspan=(xlo[sel], xhi[sel])))
        else:
            z = np.zeros(len(sel), dtype=np.int64)
            parts.append(_frame(env, C, (k0[sel], k1[sel]), a[sel], b[sel], z, z,
                                "WASTE", g, own=True, xspan=(xlo[sel], xhi[sel])))
    return (pd.concat(parts, ignore_index=True) if parts
            else _block_frame([], env["cat_cols"], sy, sz))

def _rect_rows(env, A, blk, frac, kx, iyq, lb, ins_p, extras,
               ore_blocks, bids, oz, geo) -> pd.DataFrame:
    x0, y0, dx, gq, zb0, zb1 = geo
    cat_cols, sy, sz = env["cat_cols"], env["sy"], env["sz"]
    if not len(blk):
        return _block_frame([], cat_cols, sy, sz)
    gid = A["gid"][blk]
    tp = A["tonnes"][blk] * frac
    vp = A["vol"][blk] * frac
    ip = np.where(ins_p, tp, 0.0)
    o = np.lexsort((kx, lb, iyq, gid))
    g_, r_, l_, k_, b_ = gid[o], iyq[o], lb[o], kx[o], blk[o]
    new = np.ones(len(o), dtype=bool)
    new[1:] = ((g_[1:] != g_[:-1]) | (r_[1:] != r_[:-1]) | (l_[1:] != l_[:-1])
               | (k_[1:] > k_[:-1] + 1))
    st = np.flatnonzero(new)
    runs = dict(gid=g_[st], row=r_[st], lb=l_[st], blk=b_[st], k0=k_[st],
                k1=np.maximum.reduceat(k_, st), t=np.add.reduceat(tp[o], st),
                v=np.add.reduceat(vp[o], st), i=np.add.reduceat(ip[o], st))

    o2 = np.lexsort((runs["row"], runs["k1"], runs["k0"], runs["lb"], runs["gid"]))
    R = {k: v[o2] for k, v in runs.items()}
    new = np.ones(len(o2), dtype=bool)
    new[1:] = ((R["gid"][1:] != R["gid"][:-1]) | (R["lb"][1:] != R["lb"][:-1])
               | (R["k0"][1:] != R["k0"][:-1]) | (R["k1"][1:] != R["k1"][:-1])
               | (R["row"][1:] != R["row"][:-1] + 1))
    st = np.flatnonzero(new)
    bk = R["blk"][st]
    t = np.add.reduceat(R["t"], st)
    v = np.add.reduceat(R["v"], st)
    lbr = R["lb"][st]
    xa = np.maximum(A["xmin"][bk], x0 + R["k0"][st] * dx)
    xb = np.minimum(A["xmax"][bk], x0 + (R["k1"][st] + 1) * dx)
    ya = np.maximum(A["ymin"][bk], y0 + R["row"][st] * gq)
    yb = np.minimum(A["ymax"][bk], y0 + (np.maximum.reduceat(R["row"], st) + 1) * gq)
    za = np.maximum(A["zmin"][bk], zb0)
    zb = np.minimum(A["zmax"][bk], zb1)
    ore = lbr >= 0
    grade = A["grade"][bk]
    cat = np.zeros((len(st), len(cat_cols)))
    cat[np.arange(len(st)), A["cat_idx"][bk]] = t
    cols = {
        "BLOCK_ID": np.where(ore, bids[np.maximum(lbr, 0)] if len(bids) else 0, 0),
        "ZONE_ID": np.where(ore, np.asarray(oz, dtype=np.int64)[np.maximum(lbr, 0)]
                            if len(oz) else 0, 0),
        "MATERIAL": np.where(ore, "ORE", "WASTE"),
        "X_FROM": xa, "X_TO": xb,
        "YC": (ya + yb) / 2, "YINC": yb - ya, "ZC": (za + zb) / 2, "ZINC": zb - za,
        "TONNES": t, "VOLUME": v,
        "DENSITY": np.where(v > 0, t / np.where(v > 0, v, 1.0), 0.0),
        "AU": grade, "AU_DEF": A["def_grade"][bk], "METAL_G": t * grade,
        "OXIDE": env["oxv"][A["ox_codes"][bk]],
        "OXIDE_PCT": np.where(t > 0, 100.0, 0.0),
        "TONNES_ABOVE_COG": np.add.reduceat(R["i"], st),
        "RESCAT": env["rcv"][A["cat_idx"][bk]], "cat": cat,
    }
    for name, out_name in extras["numeric"]:
        cols[out_name] = A["xn_" + name][bk]
    for name, out_name, values, _ in extras["categorical"]:
        vv = np.empty(len(values), dtype=object)
        vv[:] = values
        cols[out_name] = vv[A["xc_" + name][bk]]
    carried = (_CARRIED + [e[1] for e in extras["numeric"]]
               + [e[1] for e in extras["categorical"]])
    for key in carried:
        cols[OWN + key] = cols[key]
    if ore.any():
        idx = np.maximum(lbr, 0)
        for key in carried:
            src = ore_blocks[key].to_numpy()[idx]
            cols[key] = np.where(ore, src, cols[key])
        cols["METAL_G"] = t * cols["AU"].astype(float)
    return _block_frame(cols, cat_cols, sy, sz)

def _world_blocks(ob: pd.DataFrame, frame) -> pd.DataFrame:
    if not len(ob):
        return ob
    u = (ob["X_FROM"].to_numpy(float) + ob["X_TO"].to_numpy(float)) / 2
    x, y = frame.to_world(u, ob["YC"].to_numpy(float))
    ob = ob.copy()
    ob["XC"], ob["YC"] = x, y
    return ob

def _process_bench(job, ctx):
    iz, A, (s_lo, s_hi) = job[:3]
    fi = job[3] if len(job) > 3 else -1
    x0, y0, z0 = ctx["origin"]
    dx, dy, sy, sz = ctx["dx"], ctx["dy"], ctx["sy"], ctx["sz"]
    nk, k_lo, ny = ctx["nk"], ctx["k_lo"], ctx["ny"]
    w_lo, w_hi = s_lo - ctx["halo"], s_hi + ctx["halo"]
    extras, ncat, nox = ctx["extras"], ctx["ncat"], ctx["nox"]
    cat_cols = ctx["cat_cols"]
    grid = extras["grid"]
    by, bz, gy = grid["by"], grid["bz"], grid["gy"]
    rr, qy, q = int(round(dy / gy)), int(round(by / gy)), ctx["yfine"]
    gq = gy / q
    regions = ctx["regions"]
    frame = regions[fi] if fi >= 0 else None
    name = frame.name if frame is not None else ctx["model_name"]
    tot = _new_tot()
    zc = z0 + (iz + 0.5) * sz
    b1 = np.arange(len(A["gid"]))

    def empty():
        return dict(ob=_block_frame([], cat_cols, sy, sz),
                    dm=_block_frame([], cat_cols, sy, sz),
                    bench=dict(ZC=zc, UNDILUTED_TONNES=0.0, UNDILUTED_METAL=0.0,
                               DILUTED_TONNES=0.0, DILUTED_METAL=0.0),
                    tot=tot, cat_ore=np.zeros(ncat), gid=A["gid"],
                    ore_frac=np.zeros(len(b1)), region=name)

    fine = A.get("fine")
    if fine is not None and fine.any():
        cb, fb = np.flatnonzero(~fine), np.flatnonzero(fine)
        yr_c, iy_c, fy_c = _explode(A["ymin"][cb], A["ymax"][cb], y0, gy)
        yr_f, iy_f, fy_f = _explode(A["ymin"][fb], A["ymax"][fb], y0, gq)
        yrep = np.concatenate([cb[yr_c], fb[yr_f]])
        iyq = np.concatenate([iy_c * q, iy_f])
        fy = np.concatenate([fy_c, fy_f])
        pfine = np.concatenate([np.zeros(len(yr_c), bool), np.ones(len(yr_f), bool)])
    else:
        yrep, iyq, fy = _explode(A["ymin"], A["ymax"], y0, gy)
        iyq = iyq * q
        pfine = np.zeros(len(yrep), dtype=bool)
    if frame is None:

        st_y = iyq // (rr * ny * q)
        keep = (st_y >= w_lo) & (st_y < w_hi)
        if not keep.all():
            yrep, iyq, fy, pfine = yrep[keep], iyq[keep], fy[keep], pfine[keep]
    if not len(yrep):
        return empty()
    b2, f2 = b1[yrep], A["fz"][yrep] * fy
    xrep, kx, fx = _explode(A["xmin"][b2], A["xmax"][b2], x0, dx)
    blk, frac = b2[xrep], f2[xrep] * fx
    iyq, pfine = iyq[xrep], pfine[xrep]

    def centres(sel):
        b_ = blk[sel]
        xa = np.maximum(A["xmin"][b_], x0 + kx[sel] * dx)
        xb = np.minimum(A["xmax"][b_], x0 + (kx[sel] + 1) * dx)
        ya = np.maximum(A["ymin"][b_], y0 + iyq[sel] * gq)
        yb = np.minimum(A["ymax"][b_], y0 + (iyq[sel] + 1) * gq)
        return (xa + xb) / 2, (ya + yb) / 2

    if frame is None:
        u0, v0, nk_f = x0, y0, nk
        if regions and pfine.any():
            fsel = np.flatnonzero(pfine)
            cx, cy = centres(fsel)
            drop = np.zeros(len(blk), dtype=bool)
            drop[fsel] = region_owner(cx, cy, regions) >= 0
            keep = ~drop
            blk, frac, kx, iyq, pfine = (blk[keep], frac[keep], kx[keep],
                                         iyq[keep], pfine[keep])
        k_cell = kx
        row = iyq // (rr * q)
    else:
        u0, v0 = 0.0, 0.0
        nk_f = max(1, int(math.ceil(frame.length / dx - _EPS)))
        cx, cy = centres(np.arange(len(blk)))
        uu, vv = frame.to_local(cx, cy)
        k_cell = np.floor(uu / dx).astype(np.int64)
        row = np.floor(vv / dy).astype(np.int64)
        strip = row // ny
        keep = ((region_owner(cx, cy, regions) == fi) & (strip >= w_lo)
                & (strip < w_hi))
        blk, frac, kx, iyq, pfine = (blk[keep], frac[keep], kx[keep], iyq[keep],
                                     pfine[keep])
        k_cell = np.clip(k_cell[keep], 0, nk_f - 1)
        row = row[keep]
    if not len(blk):
        return empty()

    ucell, cinv = np.unique(row * nk_f + k_cell, return_inverse=True)
    cinv = cinv.ravel()
    nc = len(ucell)
    tp = A["tonnes"][blk] * frac
    c_t = np.bincount(cinv, weights=tp, minlength=nc)
    c_dm = np.bincount(cinv, weights=tp * A["def_grade"][blk], minlength=nc)
    c_rm = np.bincount(cinv, weights=tp * A["grade"][blk], minlength=nc)

    af = ctx["attr_filter"]
    if af:
        n_attr = af["n"]
        sel_idx = af["selected"]
        a_code = A["attr_codes"][blk]
        c_sel_t = np.zeros(nc)
        for ai in sel_idx:
            m_ = a_code == ai
            c_sel_t += np.bincount(cinv[m_], weights=tp[m_], minlength=nc)
        unsel_cell = c_sel_t <= 0.5 * c_t
        c_dm = np.where(unsel_cell, 0.0, c_dm)
    c_cat = np.zeros((nc, ncat))
    for ci in range(ncat):
        sel = A["cat_idx"][blk] == ci
        c_cat[:, ci] = np.bincount(cinv[sel], weights=tp[sel], minlength=nc)
    c_row, c_k = ucell // nk_f, ucell % nk_f
    c_strip = c_row // ny
    value_c = c_dm - ctx["cutoff"] * c_t
    insitu = (c_t > 0) & (value_c >= -1e-12) & (c_dm > 0)

    scell, sinv = np.unique(c_strip * nk_f + c_k, return_inverse=True)
    sinv = sinv.ravel()
    s_val = (np.bincount(sinv, weights=c_dm, minlength=len(scell))
             - ctx["cutoff"] * np.bincount(sinv, weights=c_t, minlength=len(scell)))
    s_cap = np.bincount(sinv, weights=np.where(insitu, c_t, 0.0), minlength=len(scell))
    s_iy, s_k = scell // nk_f, scell % nk_f

    brk = np.ones(len(scell), dtype=bool)
    brk[1:] = (s_iy[1:] != s_iy[:-1]) | (s_k[1:] != s_k[:-1] + 1)
    starts = np.flatnonzero(brk)
    ends = np.append(starts[1:], len(scell))

    def beyond(k, v):
        uc = u0 + (k + 0.5) * dx
        if frame is None:
            return region_owner(uc, v, regions) >= 0
        wx, wy = frame.to_world(uc, v)
        bb = ctx["bbox"]
        out = (wx < bb[0]) | (wx > bb[1]) | (wy < bb[2]) | (wy > bb[3])
        return out | (region_owner(wx, wy, regions) != fi)

    ka, kb = s_k[starts], s_k[ends - 1]
    vs = v0 + (s_iy[starts] + 0.5) * sy
    if frame is None:
        rs_, re_ = ka <= k_lo, kb == nk_f - 1
        if regions:
            rs_ = rs_ | beyond(ka - 1, vs)
            re_ = re_ | beyond(kb + 1, vs)
    else:
        rs_ = (ka == 0) | beyond(ka - 1, vs)
        re_ = (kb == nk_f - 1) | beyond(kb + 1, vs)
    rs_, re_ = [bool(v) for v in rs_], [bool(v) for v in re_]
    owned_run = (s_iy[starts] >= s_lo) & (s_iy[starts] < s_hi)

    rp = dict(ctx["rp"], no_ore=frame is None and ctx["outside"] == "waste"
              and bool(regions))
    D = _dilute_cells(s_val, s_cap, starts, ends, rs_, re_, rp, tot, owned_run)
    oa_, obb, oz = D["oa"], D["ob"], D["oz"]
    nb = len(oa_)
    b_strip = s_iy[oa_]
    b_fa, b_fb = s_k[oa_], s_k[obb - 1] + 1
    b_da, b_db = s_k[D["da"]], s_k[D["db"] - 1] + 1
    b_r0 = b_strip * ny
    slab = np.full(len(scell), -1, dtype=np.int64)
    if nb:
        lens = obb - oa_
        slab[np.repeat(oa_ - np.cumsum(lens) + lens, lens) + np.arange(lens.sum())] = \
            np.repeat(np.arange(nb), lens)
    c_lab = slab[sinv]

    grown = Growth.none(nb)
    if ctx["grow"] and nb:

        rmin = int(min(c_row.min(), b_r0.min())) - 1
        rmax = int(max(c_row.max(), (b_r0 + ny - 1).max())) + 1
        kmin = int(c_k.min()) - 1
        shape = (rmax - rmin + 1, int(c_k.max()) - kmin + 2)
        ri, ki = c_row - rmin, c_k - kmin
        lab = np.full(shape, VOID, dtype=np.int64)
        lab[ri, ki] = c_lab
        er, ek = np.nonzero(lab == VOID)
        er_g, ek_g = er + rmin, ek + kmin
        vc = v0 + (er_g + 0.5) * dy
        if frame is None:
            free = ((ek_g < k_lo) | (ek_g >= nk_f) | (er_g < ctx["row_lo"])
                    | (er_g >= ctx["row_hi"]))
            if regions:
                free |= beyond(ek_g, vc)
        else:
            free = (ek_g < 0) | (ek_g >= nk_f) | beyond(ek_g, vc)
        lab[er[free], ek[free]] = FREE
        val = np.zeros(shape)
        val[ri, ki] = value_c
        cap = np.zeros(shape)
        cap[ri, ki] = np.where(insitu, c_t, 0.0)
        grown = grow_blocks(lab, val, cap, b_r0 - rmin, b_fa - kmin, b_fb - kmin,
                            b_da - kmin, b_db - kmin, ny, rp["m"], ctx["ey"],
                            rp["rule"])
        c_lab = lab[ri, ki]
        mine = np.arange(s_lo * ny, s_hi * ny) - rmin
        tot["narrow_waste"] = narrow_waste(
            lab, rp["m"], ny, mine[(mine >= 0) & (mine < shape[0])])

    owned_blk = (b_strip >= s_lo) & (b_strip < s_hi)
    own_idx = np.flatnonzero(owned_blk)
    n_own = len(own_idx)
    newid = np.full(nb, -1, dtype=np.int64)
    newid[own_idx] = np.arange(n_own)
    c_ore = c_lab >= 0
    c_own_strip = (c_strip >= s_lo) & (c_strip < s_hi)
    c_emit_ore = c_ore & owned_blk[np.maximum(c_lab, 0)] if nb else c_ore
    pl = c_lab[cinv]
    p_emit = (c_emit_ore | (~c_ore & c_own_strip))[cinv]
    p_ins = insitu[cinv]
    lab_own = np.where(pl >= 0, newid[np.maximum(pl, 0)] if nb else -1, -1)
    bids = np.arange(1, n_own + 1)
    zone_own = oz[own_idx] - D["zone0"]
    tot["grown"] = int(((grown.up + grown.dn)[own_idx] > 0).sum())
    tot["elos_x"] = int(((b_fb - b_fa) > (b_db - b_da))[own_idx].sum())
    tot["elos_y"] = int(((grown.e_up + grown.e_dn)[own_idx] > 0).sum())

    env = dict(x0=u0, dx=dx, sy=sy, sz=sz, cat_cols=cat_cols,
               oxv=ctx["oxv"], rcv=ctx["rcv"])
    sel = p_emit & (pl >= 0)
    inv_b = lab_own[sel]
    _, Sb = _aggregate(A, blk[sel], frac[sel], inv_b, n_own, extras, ncat, nox)
    Sb["i"] = _cum(np.bincount(inv_b, weights=np.where(p_ins[sel], tp[sel], 0.0),
                               minlength=n_own))
    f_lo = (b_r0 - grown.dn - grown.e_dn)[own_idx]
    f_hi = (b_r0 + ny + grown.up + grown.e_up)[own_idx]
    ore_blocks = _frame(env, Sb, (b_fa[own_idx], b_fb[own_idx]), np.arange(n_own),
                        np.arange(1, n_own + 1), zone_own, bids, "ORE",
                        (v0 + (f_lo + f_hi) * (dy / 2), (f_hi - f_lo) * dy, zc, sz))
    if frame is not None:
        ore_blocks = _world_blocks(ore_blocks, frame)
    if regions:
        ore_blocks["REGION"] = name
        ore_blocks["AZIMUTH"] = round(frame.azimuth, 6) if frame is not None else 90.0

    bp = pl[sel]
    prow, pk = c_row[cinv[sel]], c_k[cinv[sel]]
    d_lo, d_hi = b_r0 - grown.dn, b_r0 + ny + grown.up
    ind = ((prow >= d_lo[bp]) & (prow < d_hi[bp]) & (pk >= b_da[bp])
           & (pk < b_db[bp])) if len(bp) else np.zeros(0, dtype=bool)
    in_grow = ind & ((prow < b_r0[bp]) | (prow >= b_r0[bp] + ny)) if len(bp) else ind
    tot["t_grown"] += float(tp[sel][in_grow].sum())
    if rp["elos"] or ctx["ey"]:
        tsel = tp[sel]
        t_d = np.bincount(inv_b[ind], weights=tsel[ind], minlength=n_own)
        dm_d = np.bincount(inv_b[ind], weights=(tsel * A["def_grade"][blk[sel]])[ind],
                           minlength=n_own)
        ore_blocks[OWN + "AU_DEF_DESIGN"] = np.where(
            t_d > 0, dm_d / np.where(t_d > 0, t_d, 1.0), 0.0)
        ore_blocks["ELOS_TONNES"] = ore_blocks["TONNES"].to_numpy(float) - t_d
        tot["t_elos"] += float((ore_blocks["TONNES"].to_numpy(float) - t_d).sum())

        out_y = ((prow < d_lo[bp]) | (prow >= d_hi[bp])) if len(bp) else ind
        tot["t_elos_y"] += float(tsel[out_y].sum())
        tot["t_elos_x"] += float(tsel[~ind & ~out_y].sum())

    dm = _block_frame([], cat_cols, sy, sz)
    if not extras.get("summary_only"):
        parts = []
        coarse = ~pfine & p_emit
        if coarse.any():
            parts.append(_bench_model(
                env, A, blk[coarse], frac[coarse], row[coarse], iyq[coarse],
                kx[coarse], A["zp"][blk[coarse]], extras, ncat, nox, ore_blocks,
                bids, zone_own, lab_own[coarse], p_ins[coarse],
                (y0, z0, dy, ny, sz, by, bz, ctx["bx"], dx, iz, qy * q, nk_f)))
        f_ = pfine & p_emit
        if f_.any():
            parts.append(_rect_rows(
                env, A, blk[f_], frac[f_], kx[f_], iyq[f_], lab_own[f_], p_ins[f_],
                extras, ore_blocks, bids, zone_own,
                (x0, y0, dx, gq, z0 + iz * sz, z0 + (iz + 1) * sz)))
        parts = [f for f in parts if len(f)]
        if parts:
            dm = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

    ore_frac = np.bincount(blk[p_emit], weights=frac[p_emit] * (pl[p_emit] >= 0),
                           minlength=len(b1))
    t_e = tp[p_emit]
    tot["rc_pieces"] = np.array([p_emit.sum(), (A["vol"][blk] * frac)[p_emit].sum(),
                                 t_e.sum(), (t_e * A["grade"][blk[p_emit]]).sum()])
    c_emit = c_emit_ore | (~c_ore & c_own_strip)
    c_v = np.bincount(cinv, weights=A["vol"][blk] * frac, minlength=nc)
    tot["rc_cells"] = np.array([c_emit.sum(), c_v[c_emit].sum(), c_t[c_emit].sum(),
                                c_rm[c_emit].sum()])
    if len(dm):
        g = (dm["X_TO"] - dm["X_FROM"]).to_numpy(float) * dm["YINC"] * dm["ZINC"]
        t_g = g.to_numpy(float) * dm["DENSITY"].to_numpy(float)
        tot["rc_model"] = np.array([len(dm), g.sum(), t_g.sum(),
                                    (t_g * dm["AU"].to_numpy(float)).sum()])
    o = c_emit_ore
    ins = insitu & c_own_strip
    lost = ins & ~c_ore
    dil = o & ~insitu
    oo = ins & c_ore
    ww = ~insitu & ~c_ore & c_own_strip
    row_ = dict(ZC=zc,
                UNDILUTED_TONNES=c_t[ins].sum(), UNDILUTED_METAL=c_rm[ins].sum(),
                DILUTED_TONNES=c_t[o].sum(), DILUTED_METAL=c_rm[o].sum(),
                ORE_ORE_TONNES=c_t[oo].sum(), ORE_ORE_METAL=c_rm[oo].sum(),
                ORE_WASTE_TONNES=c_t[lost].sum(), ORE_WASTE_METAL=c_rm[lost].sum(),
                WASTE_ORE_TONNES=c_t[dil].sum(), WASTE_ORE_METAL=c_rm[dil].sum(),
                WASTE_WASTE_TONNES=c_t[ww].sum(), WASTE_WASTE_METAL=c_rm[ww].sum())
    tot["t_in"] += row_["UNDILUTED_TONNES"]
    tot["m_in"] += row_["UNDILUTED_METAL"]
    tot["t_ore"] += row_["DILUTED_TONNES"]
    tot["m_ore"] += row_["DILUTED_METAL"]
    tot["t_dil"] += c_t[dil].sum()
    tot["m_dil"] += c_rm[dil].sum()
    tot["t_lost"] += c_t[lost].sum()
    tot["m_lost"] += c_rm[lost].sum()
    tot["t_oo"] += row_["ORE_ORE_TONNES"]
    tot["m_oo"] += row_["ORE_ORE_METAL"]
    tot["t_ww"] += row_["WASTE_WASTE_TONNES"]
    tot["m_ww"] += row_["WASTE_WASTE_METAL"]

    if af:
        tot["pd_t_in"] += row_["UNDILUTED_TONNES"]
        tot["pd_m_in"] += row_["UNDILUTED_METAL"]
        tot["pd_t_ore"] += row_["DILUTED_TONNES"]
        tot["pd_m_ore"] += row_["DILUTED_METAL"]
        tot["pd_t_dil"] += c_t[dil].sum()
        tot["pd_m_dil"] += c_rm[dil].sum()
        tot["pd_t_lost"] += c_t[lost].sum()
        tot["pd_m_lost"] += c_rm[lost].sum()
    cat_ore = c_cat[o].sum(axis=0)
    return dict(ob=ore_blocks, dm=dm, bench=row_, tot=tot,
                cat_ore=cat_ore, gid=A["gid"], ore_frac=ore_frac, region=name)

def _run_job(job, ctx):
    out = _process_bench(job, ctx)
    fin = ctx["finish"]
    ob = out.pop("ob")
    out["n_ob"] = len(ob)
    out["oxides"] = set(map(str, ob["OXIDE"].unique())) if len(ob) else set()
    if len(ob):
        ob = ob.assign(DILUTION_TONNES=ob["TONNES"] - ob["TONNES_ABOVE_COG"])
    out["ob"] = _encode(_ore_columns(ob, fin), False, None, {})
    dm = out.pop("dm")
    out["dm"] = None if fin["summary_only"] else _encode(
        dm, True, fin["grid"], fin["names"], fin["layout"])
    return out

def _ore_columns(ob: pd.DataFrame, fin: dict) -> pd.DataFrame:
    ob = ob.rename(columns={"AU_DEF": fin["def_name"],
                            OWN + "AU_DEF_DESIGN": fin["def_name"] + "_DESIGN"})
    return ob[[c for c in ob.columns if not c.startswith(OWN)]]

JOB_CELLS = 1_000_000

HALO_STRIPS = 6

def _strip_groups(s0: np.ndarray, s1: np.ndarray, cells: np.ndarray,
                  cap: int) -> list[tuple[int, int]]:
    first, last = int(s0.min()), int(s1.max())
    load = np.bincount(s0 - first, weights=cells, minlength=last - first + 1)
    groups, start, acc = [], first, 0.0
    for i, w in enumerate(load):
        s = first + i
        if acc > 0 and acc + w > cap:
            groups.append((start, s))
            start, acc = s, 0.0
        acc += w
    groups.append((start, last + 1))
    return groups

def _fine_blocks(F: dict, loc: np.ndarray, grid: dict, regions: list) -> np.ndarray:
    if not regions:
        return np.zeros(len(loc), dtype=bool)
    bx, by, x0, y0 = grid["bx"], grid["by"], grid["x0"], grid["y0"]
    xc = (F["xmin"][loc] + F["xmax"][loc]) / 2
    yc = (F["ymin"][loc] + F["ymax"][loc]) / 2
    pi = np.floor((xc - x0) / bx).astype(np.int64)
    pj = np.floor((yc - y0) / by).astype(np.int64)
    up, inv = np.unique(np.stack([pi, pj], axis=1), axis=0, return_inverse=True)
    px0, py0 = x0 + up[:, 0] * bx, y0 + up[:, 1] * by
    hit = np.zeros(len(up), dtype=bool)
    for r in regions:
        hit |= r.touches_boxes(px0, px0 + bx, py0, py0 + by)
    return hit[inv.ravel()]

def _inside_any(regions: list, xmin, xmax, ymin, ymax) -> np.ndarray:
    out = np.zeros(len(xmin), dtype=bool)
    for r in regions:
        inside = np.ones(len(xmin), dtype=bool)
        for cx in (xmin, xmax):
            for cy in (ymin, ymax):
                uu, vv = r.to_local(cx, cy)
                inside &= ((uu >= -1e-9) & (uu <= r.length + 1e-9)
                           & (vv >= -1e-9) & (vv <= r.width + 1e-9))
        out |= inside
    return out

class _Pool:

    def __init__(self, threads: int):
        self.threads = threads
        self.pool: Optional[ProcessPoolExecutor] = None
        self.broken = False

    def get(self) -> Optional[ProcessPoolExecutor]:
        if self.threads <= 1 or self.broken:
            return None
        if self.pool is None:

            self.pool = ProcessPoolExecutor(
                max_workers=self.threads,
                mp_context=multiprocessing.get_context("spawn"))
        return self.pool

    def reset(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
        self.pool = None

    def close(self, wait: bool = True) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=wait, cancel_futures=True)
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

def worker_pool(threads: int) -> _Pool:
    threads = max(1, int(threads or 1))
    if sys.platform == "win32" and threads > 61:

        log.warning("Windows allows at most 61 worker processes; using 61 "
                    "instead of %d", threads)
        threads = 61
    return _Pool(threads)

def run_dilution(p: DilutionParams,
                 progress: Optional[Callable[[str], None]] = None,
                 df=None,
                 bench_progress: Optional[Callable[[int, int], None]] = None,
                 cancel=None, summary_only: bool = False,
                 store: Optional[BlockStore] = None,
                 job_cells: Optional[int] = None,
                 pool: Optional[_Pool] = None) -> DilutionResult:
    say = progress or (lambda s: log.info(s))
    p.validate()
    if store is None:
        store = as_store(df, p, progress=say)
    _check_columns(store.columns, p)
    bx, by, bz = infer_block_size(store, p)
    sx, sy, sz = p.smu_x, p.smu_y or by, p.smu_z or bz
    gx, gy, gz = (check_smu_ratio(axis, smu, parent) for axis, smu, parent in
                  (("X", sx, bx), ("Y", sy, by), ("Z", sz, bz)))
    dx = p.step_x
    if not (abs(bx / dx - round(bx / dx)) < _TOL or abs(dx / bx - round(dx / bx)) < _TOL):
        raise ValueError(f"X step ({dx} m) must divide the parent cell X ({bx} m) "
                         "or be a multiple of it, so diluted shapes sub-cell the model")
    m = int(round(sx / dx))
    e_cells = int(round((p.elos or 0) / dx))
    dy = p.step_y or sy
    if abs(sy / dy - round(sy / dy)) > _TOL:
        raise ValueError(f"SMU Y ({sy} m) must be a whole number of Y steps ({dy} m)")
    ny = int(round(sy / dy))
    elos_y = (p.elos or 0) if p.elos_axes == "xy" else 0.0
    ey = int(round(elos_y / dy))
    if abs(elos_y / dy - ey) > _TOL:
        raise ValueError(f"Edge dilution ({p.elos} m) must be a whole number of "
                         f"Y steps ({dy} m)")

    gy = common_cell(dy, gy)
    if gy <= 0:
        raise ValueError(f"Y step ({dy} m) and the parent cell Y ({by} m) have no "
                         f"common sub-cell size of at least {by / 100:g} m")

    grow = p.elos_axes == "xy" and (ny > 1 or ey > 0)
    regions = order_regions(as_regions(p.regions)) if p.regions else []
    say(f"Parent cell {bx} x {by} x {bz} m; SMU {sx} x {sy} x {sz} m; "
        f"X step {dx} m ({m} steps per SMU); Y step {dy} m ({ny} steps per SMU); "
        f"mode {p.mode}; cut-off {p.cutoff}; "
        f"ore rule {ORE_RULES[p.ore_rule]}"
        + (f"; edge dilution {p.elos:g} m per side ({e_cells} X steps, {ey} Y steps)"
           if e_cells else ""))

    grid = model_grid(store, p, (bx, by, bz))
    x0, y0, z0 = grid["x0"], grid["y0"], grid["z0"]
    npx, npy, npz = grid["npx"], grid["npy"], grid["npz"]
    nk = int(math.ceil((grid["xhi"] - x0) / dx - _EPS))

    k_lo = int(math.floor((grid["xlo"] - x0) / dx + _EPS))
    say(f"Parent grid origin ({x0}, {y0}, {z0}); {npx} x {npy} x {npz} parent cells")
    if gy < min(dy, by) - _TOL or gz < min(sz, bz) - _TOL:
        say(f"SMU crosses parent boundaries: diluted cells are split on them "
            f"(common sub-cell Y {gy:g} m, Z {gz:g} m)")
    q = 1
    if regions:
        res_ = p.sample_size or min(dx, dy / 2)
        q = max(1, int(math.ceil(gy / res_ - 1e-9)))
        for r in regions:
            say(f"Region {r.describe()}")
        say(f"Blocks where a region is cut into {dx:g} x {gy / q:g} m pieces; "
            + ("material outside every region is diluted along the model X axis"
               if p.outside == "model" else
               "material outside every region is left as waste"))

    scan = _scan_attributes(store, p)
    categories, ox_values = scan["categories"], scan["ox_values"]
    ncat, nox = len(categories), len(ox_values)
    cat_cols = [f"TONNES_RESCAT_{_cat_label(c)}" for c in categories]
    extras = _extra_attributes(store, p)
    attr_filter = {}
    if p.dilute_attr_col and p.dilute_attr_values:
        ac = p.dilute_attr_col
        codes_arr, attr_vals = store.codes(ac)
        attr_vals = list(attr_vals) + ["MISSING"]
        sel_idx = {i for i, v in enumerate(attr_vals)
                   if v in p.dilute_attr_values}
        attr_filter = {"n": len(attr_vals), "selected": sel_idx}
        say(f"Partial dilution: cells with >50% {ac} in "
            f"{sorted(p.dilute_attr_values)} can be ore")
    names = diluted_names(p, extras)
    layout = None if p.audit_fields else dm_layout(p, extras, store.columns)
    extras["summary_only"] = summary_only
    extras["grid"] = dict(by=by, bz=bz, gy=gy, gz=gz)
    extras["rescat_values"] = [int(c) if float(c).is_integer() else c
                               for c in categories]
    oxv = np.empty(len(ox_values), dtype=object)
    oxv[:] = ox_values
    rcv = np.empty(len(extras["rescat_values"]), dtype=object)
    rcv[:] = extras["rescat_values"]
    grid_def = ((x0, y0, z0), (bx, by, bz), (npx, npy, npz))
    ob_columns = list(_block_frame([], cat_cols, sy, sz).rename(
        columns={"AU_DEF": def_col(p)}).columns)
    if regions:
        ob_columns += ["REGION", "AZIMUTH"]
    if e_cells:
        ob_columns += [def_col(p) + "_DESIGN", "ELOS_TONNES"]
    ctx = dict(origin=(x0, y0, z0), dx=dx, dy=dy, ny=ny, ey=ey, grow=grow,
               halo=HALO_STRIPS if grow else 0,
               row_lo=int(math.floor((grid["ylo"] - y0) / dy + _EPS)),
               row_hi=int(math.ceil((grid["yhi"] - y0) / dy - _EPS)),
               sy=sy, sz=sz, nk=nk, k_lo=k_lo, bx=bx,
               cutoff=p.cutoff, ncat=ncat, nox=nox, cat_cols=cat_cols, oxv=oxv,
               rcv=rcv, extras=extras, regions=regions, yfine=q,
               outside=p.outside, model_name=DEFAULT_NAME,
               bbox=(grid["xlo"], grid["xhi"], grid["ylo"], grid["yhi"]),
               rp=dict(m=m, elos=e_cells, rule=p.ore_rule),
               attr_filter=attr_filter)
    ctx["finish"] = dict(grid=grid_def, names=names, layout=layout,
                         def_name=def_col(p), summary_only=summary_only)
    field_keys = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "vol", "tonnes",
                  "grade", "def_grade", "cat_idx", "ox_codes"]
    field_keys += ["xn_" + c for c, _ in extras["numeric"]]
    field_keys += ["xc_" + e[0] for e in extras["categorical"]]
    if p.dilute_attr_col:
        field_keys.append("attr_codes")

    if layout is None:
        empty_columns = list(_add_parent_grid(_block_frame([], cat_cols, sy, sz),
                                              *grid_def).rename(columns=names).columns)
    else:
        empty_columns = [out for _, out in layout[0]] + [out for _, out in layout[1]]
    spool = None
    if not summary_only:
        spool = DilutedSpool(make_work_dir(store.work_dir))

    rz, qz, ry = int(round(sz / gz)), int(round(bz / gz)), int(round(sy / gy))
    iz_lo = int(math.floor((grid["zlo"] - z0) / sz + _EPS))
    iz_hi = int(math.ceil((grid["zhi"] - z0) / sz - _EPS))
    nbench = max(0, iz_hi - iz_lo)
    cap = int(job_cells or JOB_CELLS)
    halo = ctx["halo"]

    def jobs():
        for iz in range(iz_lo, iz_hi):
            rows = rows_in_z(store, p, (bx, by, bz), z0 + iz * sz - gz,
                             z0 + (iz + 1) * sz + gz)
            if not len(rows):
                yield iz, None, None
                continue
            F = _block_fields(store, p, rows, (bx, by, bz), categories, extras)
            solid = np.flatnonzero(~F["void"])

            zrep, izf, fzs = _explode(F["zmin"][solid], F["zmax"][solid], z0, gz)
            keep = izf // rz == iz
            loc = solid[zrep[keep]]
            if not len(loc):
                yield iz, None, None
                continue
            zp, fz = izf[keep] // qz, fzs[keep]
            fine = _fine_blocks(F, loc, grid, regions)
            xlo_, xhi_ = F["xmin"][loc], F["xmax"][loc]
            ylo, yhi = F["ymin"][loc], F["ymax"][loc]
            xcells = np.maximum(np.ceil((xhi_ - x0) / dx - _EPS)
                                - np.floor((xlo_ - x0) / dx + _EPS), 1)

            def arrays(sel):
                b = loc[sel]
                out = {k: F[k][b] for k in field_keys}
                out.update(gid=rows[b], fz=fz[sel], zp=zp[sel])
                if regions:
                    out["fine"] = fine[sel]
                return out

            sel0 = np.arange(len(loc))
            if regions:
                sel0 = np.flatnonzero(~_inside_any(regions, xlo_, xhi_, ylo, yhi))
            if len(sel0):
                s0 = np.floor((ylo[sel0] - y0) / gy + _EPS).astype(np.int64)
                s1 = np.ceil((yhi[sel0] - y0) / gy - _EPS).astype(np.int64)
                s1 = np.maximum(s1, s0 + 1)
                ncell = (s1 - s0) * xcells[sel0] * np.where(fine[sel0], q, 1)
                s0, s1 = s0 // ry, (s1 - 1) // ry
                groups = _strip_groups(s0, s1, ncell, cap)
                for lo_s, hi_s in groups:
                    sel = (sel0[(s1 >= lo_s - halo) & (s0 < hi_s + halo)]
                           if len(groups) > 1 else sel0)
                    yield iz, arrays(sel), (lo_s, hi_s), -1
            for fi, r in enumerate(regions):
                sel = r.touches_boxes(xlo_, xhi_, ylo, yhi)
                if fi:
                    sel &= ~_inside_any(regions[:fi], xlo_, xhi_, ylo, yhi)
                sel = np.flatnonzero(sel)
                if not len(sel):
                    continue
                vs = np.stack([r.to_local(cx, cy)[1]
                               for cx in (xlo_[sel], xhi_[sel])
                               for cy in (ylo[sel], yhi[sel])])
                ns = max(1, int(math.ceil(r.width / sy - _EPS)))
                s0 = np.clip(np.floor(vs.min(axis=0) / sy).astype(np.int64), 0, ns - 1)
                s1 = np.clip(np.floor(vs.max(axis=0) / sy).astype(np.int64), 0, ns - 1)
                rows_y = np.maximum(np.ceil((yhi[sel] - ylo[sel]) / (gy / q) - _EPS), 1)
                groups = _strip_groups(s0, s1, xcells[sel] * rows_y, cap)
                for lo_s, hi_s in groups:
                    sub = (sel[(s1 >= lo_s - halo) & (s0 < hi_s + halo)]
                           if len(groups) > 1 else sel)
                    yield iz, arrays(sub), (lo_s, hi_s), fi
            yield iz, None, None

    ore_frac = np.zeros(store.n)
    ob_spool = OreSpool(spool.dir if spool is not None else make_work_dir(store.work_dir),
                        ob_columns)
    if spool is None:
        weakref.finalize(ob_spool, shutil.rmtree, os.path.dirname(ob_spool.path), True)
    bench_rows = []
    n_blocks = 0
    cat_ore = np.zeros(ncat)
    tot = _new_tot()
    by_region: dict = {}
    state = dict(benches=0, jobs=0)

    def merge(out):
        nonlocal n_blocks, cat_ore
        zone_off, bid_off = tot["zones"], n_blocks
        ob_spool.add(out["ob"], bid_off, zone_off, out["oxides"])
        if spool is not None:
            spool.add(out["dm"], bid_off, zone_off)
        n_blocks += out["n_ob"]
        for k, v in out["tot"].items():
            tot[k] += v
        rt = by_region.setdefault(out["region"], dict(_new_tot(), blocks=0))
        for k, v in out["tot"].items():
            rt[k] += v
        rt["blocks"] += out["n_ob"]
        bench_rows.append(out["bench"])
        cat_ore = cat_ore + out["cat_ore"]
        np.add.at(ore_frac, out["gid"], out["ore_frac"])
        state["jobs"] += 1

    def bench_done():
        state["benches"] += 1
        if bench_progress:
            bench_progress(state["benches"], nbench)

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise DilutionCancelled("Run cancelled")

    own_pool = pool is None
    if own_pool:
        pool = worker_pool(p.threads)
    threads = pool.threads
    say(f"Processing {nbench} benches of {sz} m on {threads} thread(s), "
        f"at most {cap:,} cells per job")
    todo = jobs()
    pending: deque = deque()
    try:
        ex = None
        try:
            ex = pool.get()
        except (OSError, RuntimeError, ValueError) as e:
            pool.broken = True
            log.warning("Worker processes could not start (%s: %s); running on "
                        "1 thread", type(e).__name__, e)
        if ex is not None:

            try:
                def top_up():
                    while len(pending) < 3 * threads:
                        job = next(todo, None)
                        if job is None:
                            return
                        pending.append((job, None if job[1] is None
                                        else ex.submit(_run_job, job, ctx)))
                top_up()
                while pending:
                    job, fut = pending[0]
                    if fut is not None:
                        while True:
                            check_cancel()
                            try:
                                out = fut.result(timeout=0.2)
                                break
                            except FuturesTimeout:
                                continue
                        merge(out)
                    else:
                        bench_done()
                    pending.popleft()
                    top_up()
            except DilutionCancelled:
                for _, fut in pending:
                    if fut is not None:
                        fut.cancel()
                raise
            except Exception as e:
                log.warning("Parallel run stopped (%s: %s); finishing the "
                            "remaining jobs on 1 thread", type(e).__name__, e)
                for _, fut in pending:
                    if fut is not None:
                        fut.cancel()
                pool.reset()
        for job in itertools.chain((j for j, _ in list(pending)), todo):
            check_cancel()
            if job[1] is None:
                bench_done()
            else:
                merge(_run_job(job, ctx))
        pending.clear()
    finally:
        if own_pool:
            pool.close(wait=False)

    say(f"Identified {tot['zones']} ore zones -> {n_blocks} SMU ore blocks "
        f"({tot['recoded']} broken-down blocks recoded to waste)")
    if grow:
        say(f"{tot['grown']} ore blocks grew across strips in Y, taking "
            f"{tot['t_grown']:,.0f} t")
    if log.isEnabledFor(logging.DEBUG):
        _log_reconciliation(_source_totals(store, p, (bx, by, bz)), tot)

    log.debug("Edge dilution %s: ELOS X %g m (%d steps), ELOS Y %g m (%d rows); "
              "growth in Y %s", ELOS_AXES[p.elos_axes], e_cells * dx, e_cells,
              ey * dy, ey, "on" if grow and ny > 1 else "off")
    log.debug("Ore blocks %d: %d with ELOS in X (%.0f t), %d grown in Y (%.0f t), "
              "%d with ELOS in Y (%.0f t)", n_blocks, tot["elos_x"], tot["t_elos_x"],
              tot["grown"], tot["t_grown"], tot["elos_y"], tot["t_elos_y"])
    if tot["narrow_waste"]:
        log.warning("%d waste pieces between ore zones are narrower than an SMU",
                    tot["narrow_waste"])
    n_merged = 0
    if spool is not None and p.merge_subcells:
        from .merge import merge_spec, merge_spool
        cols = spool.columns or empty_columns
        spec = merge_spec(cols, p, (x0, y0, z0), (bx, by, bz),
                          merge_attributes(p, extras, layout))
        n_merged = merge_spool(spool, spec)
        say(f"Merged {n_merged:,} sub-cells with identical attributes in the "
            "same parent cell")
    if spool is not None:
        say(f"Diluted model: {spool.rows:,} cells")

    summary = _summary(tot, cat_ore, categories, n_blocks, attr_filter)
    bb = pd.DataFrame(bench_rows)
    if len(bb):
        bb = bb.groupby("ZC", as_index=False).sum()
        bb["UNDILUTED_AU"] = np.divide(bb.UNDILUTED_METAL, bb.UNDILUTED_TONNES,
                                       out=np.zeros(len(bb)),
                                       where=bb.UNDILUTED_TONNES.to_numpy() > 0)
        bb["DILUTED_AU"] = np.divide(bb.DILUTED_METAL, bb.DILUTED_TONNES,
                                     out=np.zeros(len(bb)),
                                     where=bb.DILUTED_TONNES.to_numpy() > 0)
        bb["DILUTED_OZ"] = bb.DILUTED_METAL / GRAMS_PER_OZ
        bb["ORE_WASTE_OZ"] = bb.ORE_WASTE_METAL / GRAMS_PER_OZ
        counts = pd.Series(ob_spool.counts, dtype=float)
        bb["ORE_BLOCKS"] = bb.ZC.map(counts).fillna(0).astype(int)
        bb = bb[["ZC", "UNDILUTED_TONNES", "UNDILUTED_AU", "DILUTED_TONNES",
                 "DILUTED_AU", "DILUTED_OZ", "ORE_BLOCKS",
                 "ORE_ORE_TONNES", "ORE_ORE_METAL",
                 "ORE_WASTE_TONNES", "ORE_WASTE_METAL", "ORE_WASTE_OZ",
                 "WASTE_ORE_TONNES", "WASTE_ORE_METAL",
                 "WASTE_WASTE_TONNES", "WASTE_WASTE_METAL"]]
        bb = bb.sort_values("ZC", ascending=False).reset_index(drop=True)

    params = asdict(p)
    params.update(block_x=bx, block_y=by, block_z=bz, smu_y=sy, smu_z=sz, step_y=dy,
                  origin_x=x0, origin_y=y0, origin_z=z0, parent_nx=npx,
                  parent_ny=npy, parent_nz=npz)
    if regions:
        params["regions"] = [dict(name=r.name, priority=r.priority,
                                  corners=[list(c) for c in r.corners],
                                  azimuth=round(r.azimuth, 6)) for r in regions]
    stats = dict(tot, jobs=state["jobs"], diluted_cells=spool.rows if spool else 0,
                 merged_subcells=n_merged)
    return DilutionResult(summary, bb, params, stats, ore_frac=ore_frac,
                          spool=spool, store=store, grid=grid,
                          empty_columns=empty_columns, run_params=p,
                          ore_spool=ob_spool, layout=layout,
                          by_region=_region_summary(by_region, regions)
                          if regions else pd.DataFrame())

def _region_summary(by_region: dict, regions: list) -> pd.DataFrame:
    rows = []
    info = {r.name: r for r in regions}
    for name in [r.name for r in regions] + [DEFAULT_NAME]:
        t = by_region.get(name)
        if t is None:
            continue
        r = info.get(name)
        rows.append(dict(
            REGION=name, PRIORITY=r.priority if r else None,
            AZIMUTH=round(r.azimuth, 3) if r else 90.0,
            UNDILUTED_TONNES=t["t_in"],
            UNDILUTED_AU=t["m_in"] / t["t_in"] if t["t_in"] else 0.0,
            DILUTED_TONNES=t["t_ore"],
            DILUTED_AU=t["m_ore"] / t["t_ore"] if t["t_ore"] else 0.0,
            DILUTED_OZ=t["m_ore"] / GRAMS_PER_OZ,
            DILUTION_PCT=100 * t["t_dil"] / t["t_ore"] if t["t_ore"] else 0.0,
            ORE_LOSS_PCT=100 * t["t_lost"] / t["t_in"] if t["t_in"] else 0.0,
            ORE_BLOCKS=t["blocks"]))
    return pd.DataFrame(rows)

def _block_frame(rows: list[dict], cat_cols: list[str], sy: float,
                 sz: float) -> pd.DataFrame:
    cols = ["BLOCK_ID", "ZONE_ID", "MATERIAL", "X_FROM", "X_TO", "XC", "XINC", "YC", "YINC",
            "ZC", "ZINC", "TONNES", "VOLUME", "DENSITY", "AU", "AU_DEF",
            "METAL_G", "METAL_OZ", "OXIDE", "OXIDE_PCT", "RESCAT",
            "TONNES_ABOVE_COG", *cat_cols]
    if not rows:
        return pd.DataFrame(columns=cols)
    if isinstance(rows, dict):
        catm = rows["cat"]
        f = pd.DataFrame({k: v for k, v in rows.items() if k != "cat"})
        f = f.infer_objects()
    else:
        f = pd.DataFrame(rows)
        catm = np.vstack(f.pop("cat").to_numpy())
    for i, col in enumerate(cat_cols):
        f[col] = catm[:, i]
    f["XC"] = (f.X_FROM + f.X_TO) / 2
    f["XINC"] = f.X_TO - f.X_FROM
    if "YINC" not in f:
        f["YINC"] = sy
    if "ZINC" not in f:
        f["ZINC"] = sz
    f["METAL_OZ"] = f.METAL_G / GRAMS_PER_OZ
    return f[cols + [c for c in f.columns if c not in cols]]

def _add_parent_grid(f: pd.DataFrame, origin, parent, counts) -> pd.DataFrame:
    (x0, y0, z0), (bx, by, bz), (nx, ny, nz) = origin, parent, counts
    f = f.copy()
    i = np.floor((f.XC.astype(float) - x0) / bx + _EPS).astype(np.int64)
    j = np.floor((f.YC.astype(float) - y0) / by + _EPS).astype(np.int64)
    k = np.floor((f.ZC.astype(float) - z0) / bz + _EPS).astype(np.int64)
    f["IJK"] = (k * ny + j) * nx + i
    for name, val in (("XMORIG", x0), ("YMORIG", y0), ("ZMORIG", z0),
                      ("PARENT_XINC", bx), ("PARENT_YINC", by),
                      ("PARENT_ZINC", bz), ("NX", nx), ("NY", ny), ("NZ", nz)):
        f[name] = val
    return f

def _gt(t: float, metal: float) -> tuple[float, float, float]:
    return t, (metal / t if t > 0 else 0.0), metal / GRAMS_PER_OZ

def _summary(tot: dict, cat_ore: np.ndarray, categories: list,
             nblocks: int, attr_filter: dict | None = None) -> pd.DataFrame:
    rows = [
        ("Undiluted: material with definition grade >= cut-off",
         *_gt(tot["t_in"], tot["m_in"])),
        ("Diluted SMU ore blocks", *_gt(tot["t_ore"], tot["m_ore"])),
        ("  of which dilution (definition grade < cut-off)",
         *_gt(tot["t_dil"], tot["m_dil"])),
        ("Ore loss: above cut-off material left in waste",
         *_gt(tot["t_lost"], tot["m_lost"])),
        ("Classification: Ore-Ore (above COG, in ore block)",
         *_gt(tot["t_oo"], tot["m_oo"])),
        ("Classification: Ore-Waste (above COG, left in waste)",
         *_gt(tot["t_lost"], tot["m_lost"])),
        ("Classification: Waste-Ore (below COG, in ore block)",
         *_gt(tot["t_dil"], tot["m_dil"])),
        ("Classification: Waste-Waste (below COG, not in ore block)",
         *_gt(tot["t_ww"], tot["m_ww"])),
    ]
    for c, t in zip(categories, cat_ore):
        rows.append((f"Diluted ore tonnes in RESCAT {_cat_label(c)}",
                     t, np.nan, np.nan))
    if attr_filter:
        rows += [
            ("Partial dilution: selected undiluted",
             *_gt(tot["pd_t_in"], tot["pd_m_in"])),
            ("Partial dilution: selected diluted ore",
             *_gt(tot["pd_t_ore"], tot["pd_m_ore"])),
            ("Partial dilution: selected dilution",
             *_gt(tot["pd_t_dil"], tot["pd_m_dil"])),
            ("Partial dilution: selected ore loss",
             *_gt(tot["pd_t_lost"], tot["pd_m_lost"])),
        ]
    s = pd.DataFrame(rows, columns=["ITEM", "TONNES", "AU", "METAL_OZ"])
    if attr_filter and tot["pd_t_ore"]:
        s.attrs["dilution_pct"] = 100 * tot["pd_t_dil"] / tot["pd_t_ore"]
        s.attrs["ore_loss_pct"] = (100 * tot["pd_t_lost"] / tot["pd_t_in"]
                                   if tot["pd_t_in"] else 0.0)
        s.attrs["ore_loss_oz"] = tot["pd_m_lost"] / GRAMS_PER_OZ
        s.attrs["dilution_tonnes"] = tot["pd_t_dil"]
        s.attrs["metal_change_pct"] = (100 * (tot["pd_m_ore"] - tot["pd_m_in"])
                                       / tot["pd_m_in"] if tot["pd_m_in"] else 0.0)
    else:
        s.attrs["dilution_pct"] = 100 * tot["t_dil"] / tot["t_ore"] if tot["t_ore"] else 0.0
        s.attrs["ore_loss_pct"] = 100 * tot["t_lost"] / tot["t_in"] if tot["t_in"] else 0.0
        s.attrs["ore_loss_oz"] = tot["m_lost"] / GRAMS_PER_OZ
        s.attrs["dilution_tonnes"] = tot["t_dil"]
        s.attrs["metal_change_pct"] = (100 * (tot["m_ore"] - tot["m_in"]) / tot["m_in"]
                                       if tot["m_in"] else 0.0)
    s.attrs["ore_blocks"] = nblocks
    s.attrs["recoded"] = tot["recoded"]
    s.attrs["grown"] = tot.get("grown", 0)
    s.attrs["t_grown"] = tot.get("t_grown", 0.0)
    s.attrs["partial_dilution"] = bool(attr_filter)
    return s

def block_outline(r) -> list[tuple[float, float, float]]:
    az = getattr(r, "AZIMUTH", 90.0)
    if pd.isna(az) or abs(az - 90.0) < 1e-9:
        x1, x2 = r.X_FROM, r.X_TO
        y1, y2 = r.YC - r.YINC / 2, r.YC + r.YINC / 2
        return [(x1, y1, r.ZC), (x2, y1, r.ZC), (x2, y2, r.ZC), (x1, y2, r.ZC),
                (x1, y1, r.ZC)]
    a = math.radians(az)
    ux, uy = math.sin(a) * r.XINC / 2, math.cos(a) * r.XINC / 2
    vx, vy = math.cos(a) * r.YINC / 2, -math.sin(a) * r.YINC / 2
    pts = [(r.XC - ux - vx, r.YC - uy - vy), (r.XC + ux - vx, r.YC + uy - vy),
           (r.XC + ux + vx, r.YC + uy + vy), (r.XC - ux + vx, r.YC - uy + vy)]
    return [(x, y, r.ZC) for x, y in pts + pts[:1]]

def _frames(ob):
    return [ob] if isinstance(ob, pd.DataFrame) else ob

def write_surpac_strings(ob, path: str) -> None:
    today = _dt.date.today().strftime("%d-%b-%y")
    with open(path, "w") as f:
        f.write(f"ore_blocks,{today},SMU ore blocks,\n")
        f.write("0,           0.000,           0.000,           0.000,"
                "           0.000,           0.000,           0.000\n")
        for r in (r for part in _frames(ob) for r in part.itertuples(index=False)):
            desc = f"{r.BLOCK_ID},{r.AU:.3f},{r.OXIDE},{r.TONNES:.1f}"
            for x, y, z in block_outline(r):
                f.write(f"1,{y:.3f},{x:.3f},{z:.3f},{desc}\n")
            f.write("0, 0.000, 0.000, 0.000,\n")
        f.write("0, 0.000, 0.000, 0.000, END\n")

def write_dxf(ob, path: str) -> None:
    with open(path, "w") as f:
        f.write("0\nSECTION\n2\nENTITIES\n")
        for r in (r for part in _frames(ob) for r in part.itertuples(index=False)):
            layer = f"ORE_{r.OXIDE}".replace(" ", "_")
            out = ["0", "POLYLINE", "8", layer, "66", "1", "10", "0.0", "20", "0.0",
                   "30", "0.0", "70", "9"]
            for x, y, z in block_outline(r)[:-1]:
                out += ["0", "VERTEX", "8", layer, "10", f"{x:.3f}", "20", f"{y:.3f}",
                        "30", f"{z:.3f}", "70", "32"]
            out += ["0", "SEQEND", "8", layer]
            f.write("\n".join(out) + "\n")
        f.write("0\nENDSEC\n0\nEOF\n")

_DM_PARENT = ("IJK", "XMORIG", "YMORIG", "ZMORIG", "NX", "NY", "NZ",
              "PARENT_XINC", "PARENT_YINC", "PARENT_ZINC")

def _dm_names(columns: list, p: DilutionParams) -> dict:
    std = {p.x_col: "XC", p.y_col: "YC", p.z_col: "ZC",
           (p.xinc_col or "XINC"): "XINC", (p.yinc_col or "YINC"): "YINC",
           (p.zinc_col or "ZINC"): "ZINC"}
    out = {}
    used = {"IJK", "XMORIG", "YMORIG", "ZMORIG", "NX", "NY", "NZ"}
    for c in columns:
        if c in std:
            out[c] = std[c]
            used.add(std[c])
    for c in columns:
        if c in out or c.upper() in _DM_PARENT:
            continue
        name = c.upper()[:8]
        k = 1
        while name in used:
            tag = str(k)
            name = c.upper()[:8 - len(tag)] + tag
            k += 1
        used.add(name)
        out[c] = name
    return out

def write_diluted_dm(res: DilutionResult, path: str) -> dict:
    from .io_dm import DmWriter, text_width
    p = res.run_params
    spool = res.spool
    columns = (spool.columns if spool is not None and spool.columns
               else list(res.empty_columns))
    names = _dm_names(columns, p)
    keep = [c for c in columns if c in names]
    (x0, y0, z0), (bx, by, bz) = ((res.params["origin_x"], res.params["origin_y"],
                                   res.params["origin_z"]),
                                  (res.params["block_x"], res.params["block_y"],
                                   res.params["block_z"]))
    nx, ny, nz = (res.params["parent_nx"], res.params["parent_ny"],
                  res.params["parent_nz"])

    width = {c: 0 for c in keep}
    if spool is not None:
        for ch in spool.chunks():
            for c in keep:
                col = ch[c]
                if col.dtype.kind not in "fiub" and col.notna().any():
                    width[c] = max(width[c], text_width(col))
    fields = [(names[c], width[c]) for c in keep]
    zc, xc, yc = (next(c for c in keep if names[c] == k) for k in ("ZC", "XC", "YC"))
    fields.insert(0, ("IJK", 0))
    implicit = dict(XMORIG=x0, YMORIG=y0, ZMORIG=z0, NX=nx, NY=ny, NZ=nz,
                    XINC=bx, YINC=by, ZINC=bz)

    def emit(w, frames):
        f = pd.concat(frames, ignore_index=True)
        i = np.floor((f[xc].to_numpy(float) - x0) / bx + _EPS).astype(np.int64)
        j = np.floor((f[yc].to_numpy(float) - y0) / by + _EPS).astype(np.int64)
        k = np.floor((f[zc].to_numpy(float) - z0) / bz + _EPS).astype(np.int64)
        out = f[keep].rename(columns=names)
        out.insert(0, "IJK", (k * ny + j) * nx + i)
        out = out.sort_values(["IJK", "ZC", "YC", "XC"], kind="mergesort")
        w.write(out)

    with DmWriter(path, fields, double=True, implicit=implicit) as w:
        if spool is not None:
            layer, frames = None, []
            for lv in spool.levels(ascending=True):
                if not len(lv):
                    continue
                k = int(math.floor((float(lv[zc].iloc[0]) - z0) / bz + _EPS))
                if frames and k != layer:
                    emit(w, frames)
                    frames = []
                layer = k
                frames.append(lv)
            if frames:
                emit(w, frames)
    std = {"XC", "YC", "ZC", "XINC", "YINC", "ZINC"}
    return {c: n for c, n in names.items() if n != c and n not in std}

def write_diluted_csv(res: DilutionResult, path: str) -> None:
    if res.spool is not None:
        res.spool.write(path, res.empty_columns)
    else:
        pd.DataFrame(columns=res.empty_columns).to_csv(path, index=False)

def iter_diluted_dm(path: str, p: DilutionParams, rows: Optional[int] = CHUNK_ROWS):
    from .io_dm import iter_dm
    back = {"XC": p.x_col, "YC": p.y_col, "ZC": p.z_col,
            "XINC": p.xinc_col or "XINC", "YINC": p.yinc_col or "YINC",
            "ZINC": p.zinc_col or "ZINC"}
    for ch in iter_dm(path, rows=rows):
        yield ch.rename(columns=back)

def write_outputs(res: DilutionResult, out_dir: str,
                  diluted_csv: bool = True) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    j = lambda name: os.path.join(out_dir, name)
    paths = {
        "ore_blocks": j("ore_blocks.csv"),
        "diluted_model": j(diluted_model_name(res.params["mode"])),
        "strings": j("ore_blocks.str"),
        "dxf": j("ore_blocks.dxf"),
        "block_model": j("block_model_flagged.csv"),
        "summary": j("summary.csv"),
        "by_bench": j("summary_by_bench.csv"),
        "report": j("report.txt"),
        "params": j("run_parameters.json"),
    }
    res.ore_spool.write(paths["ore_blocks"])
    csv_path = paths.pop("diluted_model")
    if diluted_csv:
        write_diluted_csv(res, csv_path)
        paths["diluted_model"] = csv_path
    dm_path = os.path.splitext(csv_path)[0] + ".dm"
    try:
        renamed = write_diluted_dm(res, dm_path)
        paths["diluted_model_dm"] = dm_path
        if renamed:
            log.warning("Datamine field names are at most 8 characters; in %s: %s",
                        os.path.basename(dm_path),
                        ", ".join(f"{a} -> {b}" for a, b in renamed.items()))
    except (ValueError, OSError) as e:
        log.warning("Datamine diluted model not written: %s", e)
        if not diluted_csv:
            log.warning("Writing the diluted model as CSV instead: %s", csv_path)
            write_diluted_csv(res, csv_path)
            paths["diluted_model"] = csv_path
    write_surpac_strings(res.ore_chunks(), paths["strings"])
    write_dxf(res.ore_chunks(), paths["dxf"])
    for i, ch in enumerate(res.flagged_chunks()):
        ch.to_csv(paths["block_model"], mode="w" if i == 0 else "a",
                  header=i == 0, index=False)
    res.summary.to_csv(paths["summary"], index=False)
    res.by_bench.to_csv(paths["by_bench"], index=False)
    with open(paths["params"], "w") as f:
        json.dump(res.params, f, indent=2, default=str)
    with open(paths["report"], "w") as f:
        f.write(format_report(res))
    if res.by_region is not None and len(res.by_region):
        paths["by_region"] = j("summary_by_region.csv")
        res.by_region.to_csv(paths["by_region"], index=False)
        try:
            from .preview import region_preview
            region_preview(res.store, res.run_params, j("regions_preview.pdf"))
            paths["regions_preview"] = j("regions_preview.pdf")
        except (RuntimeError, OSError) as e:
            log.warning("Region preview PDF not written: %s", e)
    return paths

def format_report(res: DilutionResult) -> str:
    p = res.params
    s = res.summary
    lines = [
        "Open pit SMU dilution run (ore blocks grow in X and Y)",
        "=" * 60,
        f"Block model : {p['input_path']}",
        f"Mode        : {p['mode']} (definition grade zeroed for "
        f"{'RESCAT = 4' if p['mode'] == 'MII' else 'RESCAT >= 3'})",
        f"Cut-off     : {p['cutoff']}",
        f"Parent cell : {p['block_x']} x {p['block_y']} x {p['block_z']} m, "
        f"origin ({p['origin_x']}, {p['origin_y']}, {p['origin_z']}), "
        f"{p['parent_nx']} x {p['parent_ny']} x {p['parent_nz']} cells",
        f"SMU (x,y,z) : {p['smu_x']} x {p['smu_y']} x {p['smu_z']} m",
        f"X step      : {p['step_x']} m",
        f"Y step      : {p.get('step_y') or p['smu_y']} m",
    ]
    if "ore_rule" in p:
        lines.append(f"Ore rule    : {ORE_RULES[p['ore_rule']]}")
    if p.get("elos"):
        lines.append(f"Edge dilution (ELOS): {p['elos']:g} m per side, at the ends of "
                     + ("each ore zone in X only (legacy, no growth in Y)"
                        if p.get("elos_axes") == "x" else
                        "each ore zone in X and on block edges facing waste in Y"))
    if p.get("dilute_attr_col"):
        lines.append(f"Partial dilution: cells with >50% {p['dilute_attr_col']} "
                     f"in {p.get('dilute_attr_values', [])} can be ore; "
                     f"others treated as waste")
    for r in p.get("regions") or []:
        lines.append(f"Region      : {r['name']} priority {r['priority']}, "
                     f"azimuth {r['azimuth']:05.1f}")
    if p.get("regions"):
        lines.append(f"Outside     : {OUTSIDE[p['outside']]}")
    lines += [
        "",
        f"{'Item':<56}{'Tonnes':>14}{'AU':>9}{'Oz':>12}",
    ]
    for _, r in s.iterrows():
        au = "" if pd.isna(r.AU) else f"{r.AU:9.3f}"
        oz = "" if pd.isna(r.METAL_OZ) else f"{r.METAL_OZ:12,.0f}"
        lines.append(f"{r.ITEM:<56}{r.TONNES:14,.0f}{au:>9}{oz:>12}")
    scope = " (selected blocks only)" if s.attrs.get("partial_dilution") else ""
    lines += [
        "",
        f"Dilution (below cut-off tonnes / diluted tonnes): {s.attrs['dilution_pct']:.1f}%{scope}",
        f"Ore loss (above cut-off tonnes left in waste)    : {s.attrs['ore_loss_pct']:.1f}%{scope}",
        f"Ore loss (ounces)                                : {s.attrs['ore_loss_oz']:,.0f}",
        f"Dilution (tonnes)                                : {s.attrs['dilution_tonnes']:,.0f}",
        f"Metal change vs undiluted                        : {s.attrs['metal_change_pct']:+.1f}%{scope}",
        f"SMU ore blocks                                   : {s.attrs['ore_blocks']}",
        f"Broken-down blocks recoded to waste              : {s.attrs['recoded']}",
        f"Ore blocks grown across strips in Y              : {s.attrs['grown']}",
        f"Tonnes taken by growth in Y                      : {s.attrs['t_grown']:,.0f}",
    ]
    br = res.by_region
    if br is not None and len(br):
        lines += ["", f"{'Region':<16}{'Undil t':>13}{'Undil AU':>10}{'Dil t':>13}"
                      f"{'Dil AU':>9}{'Oz':>11}{'Dil %':>8}{'Loss %':>8}"]
        for r in br.itertuples(index=False):
            lines.append(f"{str(r.REGION):<16}{r.UNDILUTED_TONNES:13,.0f}"
                         f"{r.UNDILUTED_AU:10.3f}{r.DILUTED_TONNES:13,.0f}"
                         f"{r.DILUTED_AU:9.3f}{r.DILUTED_OZ:11,.0f}"
                         f"{r.DILUTION_PCT:8.1f}{r.ORE_LOSS_PCT:8.1f}")
    return "\n".join(lines) + "\n"
