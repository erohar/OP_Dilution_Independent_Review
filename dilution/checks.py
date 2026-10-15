from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import (DilutionParams, DilutionResult, _explode, block_extents,
                   check_smu_ratio, common_cell, def_col, infer_block_size,
                   model_grid, rows_in_z)
from .store import as_store

_TOL = 1e-6

@dataclass
class Check:
    status: str
    name: str
    detail: str

def _chk(results, ok, name, detail_ok, detail_bad, level="FAIL"):
    results.append(Check("PASS" if ok else level, name,
                         detail_ok if ok else detail_bad))
    return ok

def _is_int(a, tol=_TOL):
    a = np.asarray(a, dtype=float)
    return np.abs(a - np.round(a)) < tol

def params_from_settings(path: str) -> DilutionParams:
    with open(path) as f:
        s = json.load(f)
    cols = {k: (v if v not in ("", "(none)") else None)
            for k, v in s.get("columns", {}).items()}
    nums = s.get("numbers", {})

    def num(key, cast=float):
        v = str(nums.get(key, "")).strip()
        return cast(v) if v else None

    return DilutionParams(
        input_path=s["input"], output_dir=s["output"], **cols,
        default_density=num("default_density") or 2.7,
        block_x=num("block_x"), block_y=num("block_y"), block_z=num("block_z"),
        origin_x=num("origin_x"), origin_y=num("origin_y"),
        origin_z=num("origin_z"),
        cutoff=num("cutoff"), smu_x=num("smu_x"), smu_y=num("smu_y"),
        smu_z=num("smu_z"), step_x=num("step_x"), step_y=num("step_y"),
        mode=s.get("mode", "MII"),
        skip_rows=num("skip_rows", int) or 0,
        threads=num("threads", int) or 1,
        extra_numeric=list(s.get("extra_numeric", [])),
        extra_categorical=list(s.get("extra_categorical", [])),
        extra_metal=list(s.get("extra_metal", [])),
        ore_rule=s.get("ore_rule", "recover"),
        elos=num("elos") or 0.0,
        elos_axes=s.get("elos_axes", "xy"),
        regions=list(s.get("regions", [])),
        outside=s.get("outside", "model"),
        sample_size=num("sample_size"),
        audit_fields=bool(s.get("audit_fields", False)))

def check_inputs(src, p: DilutionParams, progress=None) -> list[Check]:
    r: list[Check] = []

    try:
        p.validate()
        _chk(r, True, "Parameters", f"Cut-off {p.cutoff}, SMU X {p.smu_x}, "
             f"X step {p.step_x}, Y step {p.step_y or 'SMU Y (no growth)'}, "
             f"{p.mode} run", "")
    except ValueError as e:
        _chk(r, False, "Parameters", "", str(e))
        return r

    store = as_store(src, p, progress)
    columns = set(store.columns)

    mapped = {"X": p.x_col, "Y": p.y_col, "Z": p.z_col, "AU": p.grade_col,
              "RESCAT": p.rescat_col, "DENSITY": p.density_col,
              "OXIDE": p.oxide_col, "XINC": p.xinc_col, "YINC": p.yinc_col,
              "ZINC": p.zinc_col}
    for c in list(p.extra_numeric) + list(p.extra_categorical) + list(p.extra_metal):
        mapped[f"extra {c}"] = c
    missing = [f"{k}={v}" for k, v in mapped.items() if v and v not in columns]
    if not _chk(r, not missing, "Columns present",
                "All mapped columns found", f"Not in the block model: {', '.join(missing)}"):
        return r
    numeric = [p.x_col, p.y_col, p.z_col] + [c for c in (p.xinc_col, p.yinc_col,
                                                         p.zinc_col) if c]
    bad = [c for c in numeric
           if any(np.isnan(store.num(c)[sl]).any() for sl in store.chunks())]
    if not _chk(r, not bad, "Coordinates numeric",
                "Centroids and cell sizes are all numeric",
                f"Blank or non-numeric values in: {', '.join(bad)}"):
        return r
    n = store.n
    _chk(r, n > 0, "Blocks loaded", f"{n:,} blocks", "The model is empty")

    unmapped = [c for c in ("OXIDE",) if not mapped[c]]
    _chk(r, not unmapped, "Optional fields mapped", "OXIDE is mapped",
         "OXIDE is not mapped, so ore blocks get material type NONE", "WARN")

    incs = [p.xinc_col, p.yinc_col, p.zinc_col]
    if any(incs) and not all(incs):
        _chk(r, False, "Sub-cell size columns", "",
             "Map all three of X, Y and Z cell size, or none of them")
        return r
    sub = all(incs)
    size_like = [c for c in store.columns
                 if c.upper().lstrip("_") in ("XINC", "YINC", "ZINC", "XSIZE",
                                              "YSIZE", "ZSIZE", "XLENGTH",
                                              "YLENGTH", "ZLENGTH")]
    _chk(r, sub or not size_like, "Sub-cell size columns",
         "Cell sizes mapped" if sub else "Regular model (no size columns in the model)",
         f"The model has {', '.join(size_like)} but they are not mapped. A "
         "sub-celled model needs them, or the parent cell will be wrong")

    try:
        bx, by, bz = infer_block_size(store, p)
    except ValueError as e:
        _chk(r, False, "Parent cell size", "", str(e))
        return r
    parent = (bx, by, bz)
    how = "entered" if p.block_x else ("largest cell" if sub else "centroid spacing")
    _chk(r, True, "Parent cell size", f"{bx} x {by} x {bz} m ({how}); confirm "
         "this matches your model definition", "")

    def blocks():
        for sl in store.chunks():
            yield sl, block_extents(store, p, parent, sl)

    if sub:
        nonpos = irregular = big = 0
        for _, e in blocks():
            xs, ys, zs = e["xs"], e["ys"], e["zs"]
            pos = (xs > 0) & (ys > 0) & (zs > 0)
            nonpos += int((~pos).sum())
            if pos.all():
                irregular += int((~(_is_int(bx / xs) & _is_int(by / ys)
                                    & _is_int(bz / zs))).sum())
            big += int(((xs > bx + _TOL) | (ys > by + _TOL) | (zs > bz + _TOL)).sum())
        if not _chk(r, nonpos == 0, "Cell sizes positive",
                    "All cell sizes > 0", f"{nonpos:,} cells have zero or negative size"):
            return r
        _chk(r, True, "Sub-cell sizes",
             "Every sub-cell divides the parent evenly" if not irregular else
             f"{irregular:,} sub-cells have free sizes (e.g. cut to a "
             "surface); the tool splits them by volume, so this is fine", "")
        _chk(r, not big, "No cell larger than the parent",
             "No cell exceeds the parent size",
             f"{big:,} cells are larger than the parent cell")

    try:
        grid = model_grid(store, p, parent)
    except ValueError as e:
        _chk(r, False, "Model origin", "", str(e))
        return r
    x0, y0, z0 = grid["x0"], grid["y0"], grid["z0"]
    _chk(r, True, "Model origin", f"({x0}, {y0}, {z0}); confirm this matches "
         "your model definition", "")

    off = {"X": 0, "Y": 0, "Z": 0}
    eg = ""
    fine = unaligned = 0
    xs_min = np.inf
    au_n = au_missing = au_neg = au_above = 0
    au_min, au_max = np.inf, -np.inf
    au_vals = np.empty(n)
    rc_missing = 0
    rc_vals: set = set()
    d_missing = d_zero = d_odd = 0
    d_min, d_max = np.inf, -np.inf
    ox_missing = 0
    for sl, e in blocks():

        for axis, c, size, o, b in (("X", e["X"], e["xs"], x0, bx),
                                    ("Y", e["Y"], e["ys"], y0, by),
                                    ("Z", e["Z"], e["zs"], z0, bz)):
            lo = c - size / 2
            tol = 1e-3 / b
            bad_ = (np.floor((lo - o) / b + tol)
                    != np.floor((lo + size - o) / b - tol))
            off[axis] += int(bad_.sum())
            if bad_.any() and not eg:
                i = int(np.flatnonzero(bad_)[0])
                eg = (f" For example the cell at {axis} {lo[i] + size[i] / 2:g} "
                      f"(size {size[i]:g}) spans {lo[i]:g} to {lo[i] + size[i]:g}, "
                      f"which crosses a parent boundary on a {b:g} m grid from {o:g}.")
        if sub:
            xs = e["xs"]
            fine += int((xs < p.step_x - _TOL).sum())
            unaligned += int((~_is_int((e["X"] - xs / 2 - x0) / p.step_x, 1e-4)).sum())
            if len(xs):
                xs_min = min(xs_min, float(xs.min()))
        au = store.num(p.grade_col)[sl]
        ok = ~np.isnan(au)
        au_vals[au_n:au_n + int(ok.sum())] = au[ok]
        au_n += int(ok.sum())
        au_missing += int((~ok).sum())
        au_neg += int((au[ok] < 0).sum())
        au_above += int((au[ok] >= p.cutoff).sum())
        if ok.any():
            au_min, au_max = min(au_min, float(au[ok].min())), max(au_max, float(au[ok].max()))
        rc = store.num(p.rescat_col)[sl]
        rc_missing += int(np.isnan(rc).sum())
        rc_vals.update(np.unique(rc[~np.isnan(rc)]).tolist())
        if p.density_col:
            d = store.num(p.density_col)[sl]
            d_missing += int(np.isnan(d).sum())
            d_zero += int((d <= 0).sum())
            live = d[d > 0]
            d_odd += int(((live < 1.0) | (live > 5.0)).sum())
            if len(live):
                d_min, d_max = min(d_min, float(live.min())), max(d_max, float(live.max()))
        if p.oxide_col:
            ox_missing += int((store.codes(p.oxide_col)[0][sl] < 0).sum())

    bad_axes = [f"{a}: {k:,}" for a, k in off.items() if k]
    _chk(r, not bad_axes, "Cells on the parent grid",
         "Every cell sits inside one parent cell on the grid",
         "Cells crossing parent cell boundaries (" + ", ".join(bad_axes) + ")." + eg
         + " Check the parent size and origin, and that the model is not rotated")

    dup = over = 0
    k_lo = int(np.floor((grid["zlo"] - z0) / bz))
    k_hi = int(np.ceil((grid["zhi"] - z0) / bz))
    for k in range(k_lo, k_hi + 1):
        rows = rows_in_z(store, p, parent, z0 + k * bz, z0 + (k + 1) * bz)
        if not len(rows):
            continue
        e = block_extents(store, p, parent, rows)
        pk = np.floor((e["Z"] - z0) / bz).astype(np.int64)
        mine = pk == k
        if not mine.any():
            continue
        e = {c: v[mine] for c, v in e.items()}
        key = pd.DataFrame({"x": np.round(e["X"], 4), "y": np.round(e["Y"], 4),
                            "z": np.round(e["Z"], 4)})
        dup += int(key.duplicated().sum())
        pi = np.floor((e["X"] - x0) / bx).astype(np.int64)
        pj = np.floor((e["Y"] - y0) / by).astype(np.int64)
        pv = pd.Series(e["xs"] * e["ys"] * e["zs"]).groupby([pi, pj]).sum()
        over += int((pv > bx * by * bz * (1 + 1e-6)).sum())
    _chk(r, dup == 0, "No duplicate centroids", "Every centroid is unique",
         f"{dup:,} blocks share a centroid with another block")
    msg = (f"{over:,} parent cells hold more volume than a parent cell")
    if over and not sub:
        msg += ". The model looks sub-celled: map the cell size columns"
    _chk(r, over == 0, "No overlapping cells",
         "No parent cell holds more than its own volume", msg)

    sx, sy, sz = p.smu_x, p.smu_y or by, p.smu_z or bz
    for axis, smu, par in (("X", sx, bx), ("Y", sy, by), ("Z", sz, bz)):
        try:
            check_smu_ratio(axis, smu, par)
            _chk(r, True, f"SMU {axis} vs parent", f"{smu} m against parent "
                 f"{par} m", "")
        except ValueError as e:
            _chk(r, False, f"SMU {axis} vs parent", "", str(e))
    step_ok = _is_int(bx / p.step_x) or _is_int(p.step_x / bx)
    _chk(r, bool(step_ok), "X step vs parent",
         f"X step {p.step_x} m fits the parent cell",
         f"X step {p.step_x} m neither divides nor is a multiple of the "
         f"parent X {bx} m")
    dy = p.step_y or sy
    ny = sy / dy
    _chk(r, bool(_is_int(ny)), "Y step vs SMU Y",
         f"SMU Y {sy} m is {int(round(ny))} Y steps of {dy} m"
         + ("; ore blocks don't grow in Y" if round(ny) == 1 else ""),
         f"SMU Y {sy} m is not a whole number of Y steps ({dy} m)")
    _chk(r, common_cell(dy, by) > 0, "Y step vs parent",
         f"Y step {dy} m shares a sub-cell size with the parent Y {by} m",
         f"Y step {dy} m and the parent Y {by} m have no common sub-cell size "
         f"of at least {by / 100:g} m")
    if sub:
        _chk(r, not fine, "X step vs sub-cells",
             "No sub-cell is narrower than the X step",
             f"{fine:,} sub-cells are narrower than the X step "
             f"({p.step_x} m), so they are averaged within a step. Use a "
             f"smaller step (e.g. {xs_min:g} m) for full resolution", "WARN")
        _chk(r, not unaligned, "Sub-cells on the X step grid",
             "Every sub-cell edge falls on an X step",
             f"{unaligned:,} sub-cells start between X steps and are "
             "split across steps by volume", "WARN")

    _chk(r, not au_missing, "AU present", "No missing grades",
         f"{au_missing:,} blocks have no AU; they are treated as 0", "WARN")
    _chk(r, not au_neg, "AU not negative", "No negative grades",
         f"{au_neg:,} blocks have negative AU (e.g. -99 for missing); "
         "they are treated as 0", "WARN")
    if au_n:
        _chk(r, True, "AU range", f"min {au_min:.3f}, median "
             f"{float(np.median(au_vals[:au_n])):.3f}, max {au_max:.3f}", "")
        above = au_above / au_n * 100
        _chk(r, above > 0, "Material above cut-off",
             f"{above:.1f}% of blocks are at or above cut-off {p.cutoff}",
             f"No block is at or above cut-off {p.cutoff}; there will be no ore",
             "WARN")
    del au_vals

    _chk(r, not rc_missing, "RESCAT present", "No missing RESCAT",
         f"{rc_missing:,} blocks have no RESCAT; treated as 4", "WARN")
    vals = [int(v) if float(v).is_integer() else v for v in sorted(rc_vals)]
    odd = [v for v in vals if v not in (1, 2, 3, 4)]
    zeroed = "RESCAT = 4" if p.mode == "MII" else "RESCAT >= 3"
    _chk(r, not odd, "RESCAT values",
         f"Values {vals}; {zeroed} set to zero grade for the ore definition",
         f"Values {vals} include {odd}, outside 1 to 4. In a {p.mode} run only "
         f"{zeroed} is zeroed; check these codes are handled as you expect", "WARN")

    if p.density_col:
        _chk(r, not d_missing, "DENSITY present", "No missing density",
             f"{d_missing:,} blocks have no density; default "
             f"{p.default_density} used", "WARN")
        _chk(r, d_zero == 0, "Void blocks", "No zero-density blocks",
             f"{d_zero:,} blocks have density <= 0 and are treated as pit void "
             "(waste next to them must be a full SMU)", "WARN")
        _chk(r, d_odd == 0, "Density range",
             f"{d_min:.2f} to {d_max:.2f} t/m3" if np.isfinite(d_min) else "n/a",
             f"{d_odd:,} blocks have density outside 1 to 5 t/m3", "WARN")
    else:
        _chk(r, False, "DENSITY", "", "No density column; the default "
             f"{p.default_density} t/m3 is used for every block", "WARN")

    if p.oxide_col:
        codes, values = store.codes(p.oxide_col)
        types = sorted(map(str, values))
        _chk(r, not ox_missing, "OXIDE present", f"Types: {', '.join(types)}",
             f"{ox_missing:,} blocks have no OXIDE (reported as MISSING); "
             f"types: {', '.join(types)}", "WARN")

    gaps = 0
    b_lo = int(np.floor((grid["zlo"] - z0) / sz + 1e-9))
    b_hi = int(np.ceil((grid["zhi"] - z0) / sz - 1e-9))
    for kb in range(b_lo, b_hi):
        rows = rows_in_z(store, p, parent, z0 + kb * sz, z0 + (kb + 1) * sz)
        if not len(rows):
            continue
        e = block_extents(store, p, parent, rows)
        xmin, ymin, zmin = (e["X"] - e["xs"] / 2, e["Y"] - e["ys"] / 2,
                            e["Z"] - e["zs"] / 2)
        zr, kk, _ = _explode(zmin, zmin + e["zs"], z0, sz)
        keep = kk == kb
        zr = zr[keep]
        yr, jj, _ = _explode(ymin[zr], ymin[zr] + e["ys"][zr], y0, sy)
        rows_ = zr[yr]
        strip_x = pd.DataFrame({"j": jj, "lo": xmin[rows_],
                                "hi": xmin[rows_] + e["xs"][rows_]}).sort_values(["j", "lo"])
        prev_hi = strip_x.groupby("j")["hi"].cummax().shift()
        same = strip_x["j"].eq(strip_x["j"].shift())
        gaps += int((same & (strip_x["lo"] > prev_hi + 1e-6)).sum())
    _chk(r, gaps == 0, "Gaps in X",
         "No gaps along any strip; only the model's own edges are treated as edges",
         f"{gaps:,} gaps along strips. Each is treated as a pit void, so waste "
         "next to it must be a full SMU. If these are just unmodelled waste or "
         "air outside a domain, export them (density 0 means void)", "WARN")
    return r

def check_outputs(src, p: DilutionParams, res: DilutionResult) -> list[Check]:
    r: list[Check] = []
    pr = res.params
    store = res.store if src is None and res.store is not None else as_store(src, p)
    bx, by, bz = pr["block_x"], pr["block_y"], pr["block_z"]
    x0, y0, z0 = pr["origin_x"], pr["origin_y"], pr["origin_z"]
    parent = (bx, by, bz)
    sx = p.smu_x
    frac = np.clip(res.ore_frac, 0, 1)

    t_in = m_in = t_ore_bm = 0.0
    xlo, xhi = np.inf, -np.inf
    for sl in store.chunks():
        e = block_extents(store, p, parent, sl)
        if p.density_col:
            d = store.num(p.density_col)[sl]
            d = np.where(np.isfinite(d), d, p.default_density)
            d = np.where(d > 0, d, 0.0)
        else:
            d = np.full(len(e["X"]), p.default_density)
        t = e["xs"] * e["ys"] * e["zs"] * d
        au = np.clip(np.nan_to_num(np.asarray(store.num(p.grade_col)[sl], float),
                                   nan=0.0), 0, None)
        t_in += float(t.sum())
        m_in += float((t * au).sum())
        t_ore_bm += float((t * frac[sl]).sum())
        if len(t):
            xlo = min(xlo, float((e["X"] - e["xs"] / 2).min()))
            xhi = max(xhi, float((e["X"] + e["xs"] / 2).max()))

    t_dm = m_dm = 0.0
    n_dm = straddle = 0
    levels = res.spool.levels() if res.spool is not None else []
    for dm in levels:
        dm = diluted_view(dm, p, res)
        n_dm += len(dm)
        t_dm += float(dm.TONNES.sum())
        m_dm += float(dm.METAL_G.sum())

        for c, inc, o, b in (("XC", "XINC", x0, bx), ("YC", "YINC", y0, by),
                             ("ZC", "ZINC", z0, bz)):
            lo = np.floor((dm[c] - dm[inc] / 2 - o) / b + 1e-9)
            hi = np.floor((dm[c] + dm[inc] / 2 - o) / b - 1e-9)
            straddle += int((lo != hi).sum())

    _chk(r, abs(t_dm - t_in) <= 1e-6 * max(t_in, 1), "Tonnes conserved",
         f"Diluted model {t_dm:,.0f} t = input {t_in:,.0f} t",
         f"Diluted model {t_dm:,.0f} t vs input {t_in:,.0f} t")
    _chk(r, abs(m_dm - m_in) <= 1e-6 * max(m_in, 1), "Metal conserved",
         f"Diluted model {m_dm / 31.1034768:,.0f} oz = input",
         f"Diluted model {m_dm / 31.1034768:,.0f} oz vs input "
         f"{m_in / 31.1034768:,.0f} oz")

    n_ob = n_narrow = n_wide = n_steps = n_low = n_ox = 0
    n_narrow_y = n_deep = n_steps_y = 0
    sy = pr["smu_y"]
    dy = pr.get("step_y") or sy
    cat_ok = True
    t_ore = 0.0
    elos = p.elos or 0.0
    elos_y = elos if p.elos_axes == "xy" else 0.0
    design = def_col(p) + "_DESIGN"
    for ob in res.ore_chunks():
        n_ob += len(ob)
        t_ore += float(ob.TONNES.sum())
        n_narrow += int((ob.XINC < sx - _TOL).sum())
        n_wide += int((ob.XINC > 2 * sx - p.step_x + 2 * elos + _TOL).sum())
        n_steps += int((~_is_int(ob.XINC / p.step_x, 1e-4)).sum())
        n_narrow_y += int((ob.YINC < sy - _TOL).sum())
        n_deep += int((ob.YINC > 2 * sy - dy + 2 * elos_y + _TOL).sum())
        n_steps_y += int((~_is_int(ob.YINC / dy, 1e-4)).sum())

        g = ob[design] if design in ob.columns else ob[def_col(p)]
        n_low += int((g < p.cutoff - 1e-9).sum())
        cat = [c for c in ob.columns if c.startswith("TONNES_RESCAT_")]
        cat_ok &= bool(np.allclose(ob[cat].sum(axis=1), ob.TONNES))
        n_ox += int((ob.OXIDE_PCT <= 0).sum())
    if n_ob:
        _chk(r, not n_narrow, "Ore blocks at least one SMU wide",
             f"All {n_ob:,} ore blocks are {sx} m or wider",
             f"{n_narrow:,} ore blocks are narrower than {sx} m")
        extra = f" plus edge dilution {2 * elos:g} m" if elos else ""
        _chk(r, not n_wide, "Ore blocks broken down",
             f"No ore block is wider than 2 SMU less one step{extra}",
             f"{n_wide:,} ore blocks are wider than 2 SMU less one step{extra}")
        _chk(r, not n_steps, "Ore widths in whole X steps",
             "Every ore block is a whole number of X steps",
             f"{n_steps:,} ore blocks are not whole X steps")
        _chk(r, not n_narrow_y, "Ore blocks at least one SMU deep",
             f"All {n_ob:,} ore blocks are {sy} m or deeper in Y",
             f"{n_narrow_y:,} ore blocks are shallower than {sy} m in Y")
        _chk(r, not n_deep, "Ore block growth in Y",
             f"No ore block is deeper than 2 SMU less one Y step{extra}",
             f"{n_deep:,} ore blocks are deeper than 2 SMU less one Y step{extra}")
        _chk(r, not n_steps_y, "Ore depths in whole Y steps",
             "Every ore block is a whole number of Y steps",
             f"{n_steps_y:,} ore blocks are not whole Y steps")
        before = " before edge dilution" if elos else ""
        _chk(r, not n_low, "Ore blocks at or above cut-off",
             f"Every ore block's definition grade{before} is >= {p.cutoff}",
             f"{n_low:,} ore blocks are below cut-off on definition grade{before}")
        _chk(r, cat_ok, "RESCAT tonnes add up",
             "Tonnes by RESCAT sum to each block's tonnes",
             "Tonnes by RESCAT do not sum to the block tonnes")
        _chk(r, not n_ox, "OXIDE assigned", "Every ore block has a dominant OXIDE",
             "Some ore blocks have no OXIDE mass")
    else:
        _chk(r, False, "Ore blocks", "", "No ore blocks were produced", "WARN")

    narrow = int(res.stats.get("narrow_waste", 0))
    _chk(r, narrow == 0, "Waste at least one SMU wide",
         "Waste between ore is at least one SMU wide along X on every row and "
         "one SMU deep along Y; narrower waste is only at the model edge, a "
         "region edge or (with edge dilution) against a void",
         f"{narrow:,} waste pieces between ore are narrower than an SMU "
         f"({sx} m in X, {sy} m in Y)")
    if n_dm:
        _chk(r, straddle == 0, "Diluted model on the parent grid",
             f"Every diluted cell sits in one {bx} x {by} x {bz} m parent cell",
             f"{straddle:,} diluted cells straddle parent cells")

    bb = res.by_bench
    if len(bb) and "ORE_ORE_TONNES" in bb.columns:
        oo_t = float(bb.ORE_ORE_TONNES.sum())
        ow_t = float(bb.ORE_WASTE_TONNES.sum())
        wo_t = float(bb.WASTE_ORE_TONNES.sum())
        und_t = float(bb.UNDILUTED_TONNES.sum())
        dil_t = float(bb.DILUTED_TONNES.sum())
        cls_und = oo_t + ow_t
        cls_dil = oo_t + wo_t
        _chk(r, abs(cls_und - und_t) <= 1e-6 * max(und_t, 1),
             "Classification sums to undiluted",
             f"Ore-Ore + Ore-Waste = {cls_und:,.0f} t = undiluted {und_t:,.0f} t",
             f"Ore-Ore + Ore-Waste = {cls_und:,.0f} t vs undiluted {und_t:,.0f} t")
        _chk(r, abs(cls_dil - dil_t) <= 1e-6 * max(dil_t, 1),
             "Classification sums to diluted",
             f"Ore-Ore + Waste-Ore = {cls_dil:,.0f} t = diluted {dil_t:,.0f} t",
             f"Ore-Ore + Waste-Ore = {cls_dil:,.0f} t vs diluted {dil_t:,.0f} t")

    _chk(r, bool(((frac >= -1e-9) & (frac <= 1 + 1e-9)).all()), "ORE_FRAC range",
         "ORE_FRAC is between 0 and 1 on every block",
         "ORE_FRAC falls outside 0 to 1")
    _chk(r, abs(t_ore_bm - t_ore) <= 1e-6 * max(t_ore, 1), "ORE_FRAC matches ore",
         "Flagged block model ore tonnes = ore block tonnes",
         f"Flagged model ore {t_ore_bm:,.0f} t vs ore blocks {t_ore:,.0f} t")
    return r

def diluted_view(dm: pd.DataFrame, p: DilutionParams, res: DilutionResult) -> pd.DataFrame:
    if "TONNES" in dm.columns and "METAL_G" in dm.columns:
        return dm
    cols, _ = res.layout
    name = dict(cols)
    out = pd.DataFrame({k: dm[name[k]].astype(float)
                        for k in ("XC", "YC", "ZC", "XINC", "YINC", "ZINC")})
    vol = out.XINC * out.YINC * out.ZINC
    dens = (dm[p.density_col].astype(float) if p.density_col
            else pd.Series(p.default_density, index=dm.index))
    out["TONNES"] = vol * dens
    out["METAL_G"] = out.TONNES * dm[f"D_{p.grade_col}"].astype(float)
    out["MATERIAL"] = dm["MATERIAL"]
    return out

def format_checks(checks: list[Check]) -> str:
    w = max(len(c.name) for c in checks) + 2
    lines = [f"{'Status':<7}{'Check':<{w}}Detail", "-" * (w + 60)]
    for c in checks:
        lines.append(f"{c.status:<7}{c.name:<{w}}{c.detail}")
    counts = {s: sum(c.status == s for c in checks) for s in ("PASS", "WARN", "FAIL")}
    lines += ["", f"{counts['PASS']} passed, {counts['WARN']} warnings, "
                  f"{counts['FAIL']} failed"]
    return "\n".join(lines) + "\n"
