from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("dilution")

_EPS = 1e-9
_UNIT = 1e5
_ADDITIVE = ("TONNES", "VOLUME", "METAL_G", "METAL_OZ", "TONNES_ABOVE_COG")
_ADDITIVE_PREFIX = "TONNES_RESCAT_"
_EXTENT = ("X_FROM", "X_TO")

RULES = {
    "avg_mass": "Average by mass", "avg_volume": "Average by volume",
    "avg_metal": "Average by metal", "dom_mass": "Dominant by mass",
    "dom_volume": "Dominant by volume", "dom_metal": "Dominant by metal",
}
GROUP_ON = "Group on"

@dataclass
class MergeSpec:
    xc: str = "XC"
    yc: str = "YC"
    zc: str = "ZC"
    xinc: str = "XINC"
    yinc: str = "YINC"
    zinc: str = "ZINC"
    origin: tuple = (0.0, 0.0, 0.0)
    parent: tuple = (1.0, 1.0, 1.0)
    density: Optional[str] = None
    default_density: float = 2.7
    additive: list = field(default_factory=list)
    initial: dict = field(default_factory=dict)
    rules: dict = field(default_factory=dict)
    grade: Optional[str] = None

def merge_spec(columns, p, origin, parent, initial: Optional[dict] = None) -> MergeSpec:
    cols = list(columns)
    own = {p.x_col, p.y_col, p.z_col} <= set(cols)
    spec = MergeSpec(origin=tuple(origin), parent=tuple(parent),
                     default_density=p.default_density)
    if own:
        spec.xc, spec.yc, spec.zc = p.x_col, p.y_col, p.z_col
        spec.xinc = p.xinc_col or "XINC"
        spec.yinc = p.yinc_col or "YINC"
        spec.zinc = p.zinc_col or "ZINC"
        spec.density = p.density_col
        spec.initial = {c: k for c, k in (initial or {}).items() if c in cols}
        spec.grade = p.grade_col if p.grade_col in cols else None
        spec.rules = {c: r for c, r in (getattr(p, "merge_rules", None) or {}).items()
                      if c in spec.initial}
        for c, r in spec.rules.items():
            if r not in RULES:
                raise ValueError(f"Merge rule for {c} must be one of {', '.join(RULES)}")
            if spec.initial[c] == "cat" and r.startswith("avg"):
                raise ValueError(f"{c} is a code, so it can only be combined by "
                                 "dominant mass, volume or metal")
    else:
        spec.density = "DENSITY"
    spec.additive = [c for c in cols if c in _ADDITIVE or c.startswith(_ADDITIVE_PREFIX)]
    return spec

def _bounds(f: pd.DataFrame, spec: MergeSpec):
    lo, hi = [], []
    for c, w in ((spec.xc, spec.xinc), (spec.yc, spec.yinc), (spec.zc, spec.zinc)):
        c_, w_ = f[c].to_numpy(float), f[w].to_numpy(float)
        lo.append(np.rint((c_ - w_ / 2) * _UNIT).astype(np.int64))
        hi.append(np.rint((c_ + w_ / 2) * _UNIT).astype(np.int64))
    return lo, hi

def _groups(f: pd.DataFrame, spec: MergeSpec) -> np.ndarray:
    skip = {spec.xc, spec.yc, spec.zc, spec.xinc, spec.yinc, spec.zinc,
            *spec.additive, *_EXTENT, *spec.rules}
    if spec.density:
        skip.add(spec.density)
    cols = {}
    for c in f.columns:
        if c in skip:
            continue
        v = f[c]
        if v.dtype.kind == "f":
            v = v.round(9)
        cols[c] = v
    key = pd.DataFrame(cols, index=f.index)
    (x0, y0, z0), (bx, by, bz) = spec.origin, spec.parent
    for name, c, o, b in (("_i", spec.xc, x0, bx), ("_j", spec.yc, y0, by),
                          ("_k", spec.zc, z0, bz)):
        key[name] = np.floor((f[c].to_numpy(float) - o) / b + _EPS).astype(np.int64)
    return key.groupby(list(key.columns), sort=False, dropna=False).ngroup().to_numpy()

def _weights(f: pd.DataFrame, spec: MergeSpec) -> dict:
    vol = (f[spec.xinc].to_numpy(float) * f[spec.yinc].to_numpy(float)
           * f[spec.zinc].to_numpy(float))
    if "VOLUME" in f.columns:
        vol = f["VOLUME"].to_numpy(float)
    if "TONNES" in f.columns:
        t = f["TONNES"].to_numpy(float)
    else:
        d = (f[spec.density].to_numpy(float) if spec.density in f.columns
             else np.full(len(f), spec.default_density))
        t = vol * np.where(np.isfinite(d), d, spec.default_density)
    g = f[spec.grade].to_numpy(float) if spec.grade in f.columns else np.ones(len(f))
    return {"mass": t, "volume": vol, "metal": t * np.where(np.isfinite(g), g, 0.0)}

def _combine(f: pd.DataFrame, cell: np.ndarray, rows: np.ndarray, spec: MergeSpec,
             w: dict) -> dict:
    out = {}
    cid = cell[rows]
    for col, rule in spec.rules.items():
        kind, weight = rule.split("_")
        v = f[col].iloc[rows]
        wt = w[weight][rows]
        if kind == "avg":
            x = pd.to_numeric(v, errors="coerce").to_numpy(float)
            ok = np.isfinite(x)
            num = pd.Series(np.where(ok, wt * x, 0.0)).groupby(cid).sum()
            den = pd.Series(np.where(ok, wt, 0.0)).groupby(cid).sum()
            mean = pd.Series(np.where(ok, x, 0.0)).groupby(cid).sum() / \
                pd.Series(ok.astype(float)).groupby(cid).sum().replace(0, np.nan)
            out[col] = (num / den.replace(0, np.nan)).fillna(mean)
        else:
            t = pd.DataFrame({"c": cid, "v": v.to_numpy(), "w": wt}).dropna(subset=["v"])
            tot = t.groupby(["c", "v"], sort=False)["w"].sum().reset_index()
            tot = tot.sort_values(["c", "w"], ascending=[True, False], kind="stable")
            out[col] = tot.drop_duplicates("c").set_index("c")["v"]
    return out

def merge_cells(f: pd.DataFrame, spec: MergeSpec) -> pd.DataFrame:
    n = len(f)
    if n < 2:
        return f
    f = f.reset_index(drop=True)
    lo, hi = _bounds(f, spec)
    gid = _groups(f, spec)
    first = np.arange(n)
    cell = np.arange(n)
    merged_any = False
    while True:
        changed = False
        for axis in (0, 1, 2):
            o1, o2 = [a for a in (0, 1, 2) if a != axis]
            order = np.lexsort((lo[axis], hi[o2], lo[o2], hi[o1], lo[o1], gid))
            a, b = order[1:], order[:-1]
            join = ((gid[a] == gid[b]) & (lo[o1][a] == lo[o1][b])
                    & (hi[o1][a] == hi[o1][b]) & (lo[o2][a] == lo[o2][b])
                    & (hi[o2][a] == hi[o2][b]) & (lo[axis][a] == hi[axis][b]))
            if not join.any():
                continue
            changed = merged_any = True
            flag = np.r_[True, ~join]
            start = np.flatnonzero(flag)
            new_id = np.empty(len(order), dtype=np.int64)
            new_id[order] = np.cumsum(flag) - 1
            cell = new_id[cell]
            lo = [np.minimum.reduceat(v[order], start) for v in lo]
            hi = [np.maximum.reduceat(v[order], start) for v in hi]
            gid = gid[order][start]
            first = first[order][start]
        if not changed:
            break
    if not merged_any:
        return f
    m = len(first)
    count = np.bincount(cell, minlength=m)
    many = count > 1
    out = f.iloc[first].reset_index(drop=True)
    for c, w_, k in ((spec.xc, spec.xinc, 0), (spec.yc, spec.yinc, 1),
                     (spec.zc, spec.zinc, 2)):
        out.loc[many, c] = ((lo[k] + hi[k]) / 2 / _UNIT)[many]
        out.loc[many, w_] = ((hi[k] - lo[k]) / _UNIT)[many]
    for c in spec.additive:
        out.loc[many, c] = np.bincount(cell, weights=f[c].to_numpy(float), minlength=m)[many]
    for c in (c for c in _EXTENT if c in f.columns):
        agg = f[c].groupby(cell).min() if c == "X_FROM" else f[c].groupby(cell).max()
        out.loc[many, c] = agg.to_numpy()[many]
    w = _weights(f, spec) if (spec.rules or spec.density) else {}
    if spec.rules:
        rows = np.flatnonzero(many[cell])
        for col, ser in _combine(f, cell, rows, spec, w).items():
            idx = ser.index.to_numpy()
            if out[col].dtype.kind in "iu" and ser.dtype.kind == "f":
                out[col] = out[col].astype(float)
            out.loc[idx, col] = ser.to_numpy()

    if spec.density and spec.density in out.columns:
        vol = ((hi[0] - lo[0]) / _UNIT * (hi[1] - lo[1]) / _UNIT * (hi[2] - lo[2]) / _UNIT)
        tonnes = np.bincount(cell, weights=w["mass"], minlength=m)
        dens = np.where(vol > 0, tonnes / np.where(vol > 0, vol, 1.0), 0.0)
        out.loc[many, spec.density] = dens[many]
    if "METAL_OZ" in out.columns and "METAL_G" in out.columns:
        from .core import GRAMS_PER_OZ
        out.loc[many, "METAL_OZ"] = out.loc[many, "METAL_G"] / GRAMS_PER_OZ
    return out.sort_values([spec.zc, spec.yc, spec.xc], ascending=[False, True, True],
                           kind="stable").reset_index(drop=True)

def merge_spool(spool, spec: MergeSpec) -> int:
    from .core import DilutedSpool
    z0, bz = spec.origin[2], spec.parent[2]
    work = tempfile.mkdtemp(prefix="merge_", dir=spool.dir)
    new = DilutedSpool(work)
    new.columns = spool.columns
    before = after = 0
    layer, frames = None, []

    def flush():
        nonlocal before, after
        if not frames:
            return
        f = pd.concat(frames, ignore_index=True)
        m = merge_cells(f, spec)
        before += len(f)
        after += len(m)
        new.add_frame(m, spec.zc)

    try:
        for lv in spool.levels(ascending=False):
            if not len(lv):
                continue
            k = int(math.floor((float(lv[spec.zc].iloc[0]) - z0) / bz + _EPS))
            if frames and k != layer:
                flush()
                frames = []
            layer = k
            frames.append(lv)
        flush()

        for path in spool.paths():
            os.remove(path)
        files = {}
        for key, path in new.files.items():
            dest = os.path.join(spool.dir, os.path.basename(path))
            os.replace(path, dest)
            files[key] = dest
    finally:
        new._finalizer.detach()
        shutil.rmtree(work, ignore_errors=True)
    spool.files, spool.rows = files, after
    return before - after
