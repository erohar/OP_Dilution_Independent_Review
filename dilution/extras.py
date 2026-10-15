from __future__ import annotations

import dataclasses
import datetime as _dt
import getpass
import logging
import math
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .core import (GRAMS_PER_OZ, DilutionParams, DilutionResult, clean_grade,
                   def_col, diluted_model_name, run_dilution, write_diluted_dm)

log = logging.getLogger(__name__)

HISTORY_DIR = os.path.join(os.path.expanduser("~"), ".open_pit_dilution_2d")
HISTORY_FILE = os.path.join(HISTORY_DIR, "run_history.csv")
SENS_MODEL_DIR = "sensitivity_models"
GT_STEP = 0.01

def parse_list(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        return []
    low = text.lower().replace("step", ":").replace(" to ", ":")
    if ":" in low:
        parts = [t for t in low.replace(",", " ").split(":")]
        nums = [float(t) for t in parts if t.strip()]
        if len(nums) != 3:
            raise ValueError(f"A range needs start, stop and step: '{text}'")
        return value_range(*nums)
    out = []
    for t in text.replace(",", " ").replace(";", " ").split():
        t = t.strip().strip('"')
        if t:
            out.append(float(t))
    return out

def value_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("The range step must be greater than zero")
    if stop < start:
        raise ValueError(f"Range stop ({stop}) is below its start ({start})")
    n = int(math.floor((stop - start) / step + 1e-9)) + 1
    dec = max(0, -int(math.floor(math.log10(step))) + 3)
    return [round(start + i * step, dec) for i in range(n)]

def fit_step(smu: float, base_step: float, parent: float) -> float:
    def fits(d):
        return (abs(parent / d - round(parent / d)) < 1e-6
                or abs(d / parent - round(d / parent)) < 1e-6)
    for k in range(1, 65):
        d = base_step / k
        if abs(smu / d - round(smu / d)) < 1e-6 and fits(d):
            return round(d, 9)
    n0 = max(1, math.ceil(smu / base_step - 1e-9))
    for n in range(n0, n0 + 10000):
        d = smu / n
        if fits(d):
            return round(d, 9)
    raise ValueError(f"No step fits SMU {smu} and the parent cell {parent}")

def result_row(res: DilutionResult) -> dict:
    s = res.summary
    und, dil = s.iloc[0], s.iloc[1]
    return {
        "UNDILUTED_TONNES": und.TONNES, "UNDILUTED_AU": und.AU,
        "UNDILUTED_OZ": und.METAL_OZ, "DILUTED_TONNES": dil.TONNES,
        "DILUTED_AU": dil.AU, "DILUTED_OZ": dil.METAL_OZ,
        "DILUTION_PCT": s.attrs["dilution_pct"],
        "ORE_LOSS_PCT": s.attrs["ore_loss_pct"],
        "ORE_LOSS_OZ": s.attrs["ore_loss_oz"],
        "DILUTION_TONNES": s.attrs["dilution_tonnes"],
        "METAL_CHANGE_PCT": s.attrs["metal_change_pct"],
        "ORE_BLOCKS": s.attrs["ore_blocks"],
    }

SENS_PARAMS = {"cutoff": "Cut-off", "smu_x": "SMU X", "smu_y": "SMU Y",
               "smu_z": "SMU Z", "step_x": "X step", "step_y": "Y step"}

class GradeTonnage:

    def __init__(self, step: float = GT_STEP):
        self.step = step
        self.tonnes = np.zeros(0)
        self.metal = np.zeros(0)

    def add(self, grade, tonnes) -> None:
        g = clean_grade(grade)
        t = np.nan_to_num(np.asarray(tonnes, dtype=float))
        k = np.floor(np.maximum(g, 0.0) / self.step + 1e-9).astype(np.int64)
        n = max(len(self.tonnes), int(k.max()) + 1 if len(k) else 0)
        self.tonnes = np.pad(self.tonnes, (0, n - len(self.tonnes)))
        self.metal = np.pad(self.metal, (0, n - len(self.metal)))
        self.tonnes += np.bincount(k, weights=t, minlength=n)
        self.metal += np.bincount(k, weights=t * g, minlength=n)

    def curve(self, cutoffs) -> pd.DataFrame:
        t_above = np.cumsum(self.tonnes[::-1])[::-1]
        m_above = np.cumsum(self.metal[::-1])[::-1]
        k = np.floor(np.asarray(cutoffs, dtype=float) / self.step + 1e-9).astype(np.int64)
        inside = k < len(t_above)
        kk = np.clip(k, 0, max(len(t_above) - 1, 0))
        t = np.where(inside, t_above[kk] if len(t_above) else 0.0, 0.0)
        m = np.where(inside, m_above[kk] if len(m_above) else 0.0, 0.0)
        return pd.DataFrame({"CUTOFF": cutoffs, "TONNES": t,
                             "GRADE": np.divide(m, t, out=np.zeros_like(t), where=t > 0),
                             "METAL_OZ": m / GRAMS_PER_OZ})

    def quantile(self, q: float) -> float:
        c = np.cumsum(self.tonnes)
        if not len(c) or c[-1] <= 0:
            return 0.0
        return (int(np.searchsorted(c, q * c[-1])) + 1) * self.step

def _ore_cells(ch: pd.DataFrame, p: DilutionParams) -> tuple[pd.DataFrame, np.ndarray]:
    ch = ch[ch["MATERIAL"].astype(str) == "ORE"]
    if "TONNES" in ch.columns:
        return ch, ch["TONNES"].to_numpy(float)
    inc = [p.xinc_col or "XINC", p.yinc_col or "YINC", p.zinc_col or "ZINC"]
    dens = (ch[p.density_col].to_numpy(float)
            if p.density_col and p.density_col in ch.columns else p.default_density)
    return ch, ch[inc[0]].to_numpy(float) * ch[inc[1]].to_numpy(float) \
        * ch[inc[2]].to_numpy(float) * dens

def model_grade_tonnage(res: DilutionResult) -> GradeTonnage:
    p = res.run_params
    gt = GradeTonnage()
    if res.spool is None:
        return gt
    for ch in res.spool.chunks():
        ch, t = _ore_cells(ch, p)
        gt.add(ch[f"D_{p.grade_col}"].to_numpy(), t)
    return gt

def _cell_tonnes(ch: pd.DataFrame, p: DilutionParams) -> np.ndarray:
    if "TONNES" in ch.columns:
        return np.nan_to_num(ch["TONNES"].to_numpy(float))
    inc = [p.xinc_col or "XINC", p.yinc_col or "YINC", p.zinc_col or "ZINC"]
    dens = (ch[p.density_col].to_numpy(float)
            if p.density_col and p.density_col in ch.columns else p.default_density)
    return np.nan_to_num(ch[inc[0]].to_numpy(float) * ch[inc[1]].to_numpy(float)
                         * ch[inc[2]].to_numpy(float) * dens)

def model_benches(chunks, p: DilutionParams, cutoff: float) -> pd.DataFrame:
    grade = f"D_{p.grade_col}"
    parts = []
    for ch in chunks:
        orig = p.grade_col if p.grade_col in ch.columns else def_col(p)
        for c in (orig, grade):
            if c not in ch.columns:
                raise ValueError(f"The saved model has no {c} field")
        t = _cell_tonnes(ch, p)
        part = {"ZC": ch[p.z_col].to_numpy(float)}
        for tag, col in (("UND", orig), ("DIL", grade)):
            g = clean_grade(ch[col].to_numpy())
            above = g >= cutoff - 1e-12
            part[f"{tag}_T"] = np.where(above, t, 0.0)
            part[f"{tag}_M"] = np.where(above, t * g, 0.0)
        parts.append(pd.DataFrame(part))
    if not parts:
        return pd.DataFrame(columns=["ZC", "UND_T", "UND_M", "DIL_T", "DIL_M"])
    return pd.concat(parts).groupby("ZC", sort=True).sum().reset_index()

def _spool_benches(res: DilutionResult, cutoff: float) -> pd.DataFrame:
    if res.spool is None:
        return pd.DataFrame()
    return model_benches(res.spool.chunks(), res.run_params, cutoff)

def sensitivity_model_name(mode: str, key: str, value: float) -> str:
    return f"{os.path.splitext(diluted_model_name(mode))[0]}_{key}_{value:g}.csv"

def _save_model(res: DilutionResult, path: str, csv: bool = True
                ) -> tuple[GradeTonnage, Optional[str], Optional[str]]:
    csv_path = None
    if csv:
        res.spool.write(path, res.empty_columns)
        csv_path = path
    dm_path = os.path.splitext(path)[0] + ".dm"
    try:
        renamed = write_diluted_dm(res, dm_path)
        if renamed:
            log.debug("Datamine field renames in %s: %s", os.path.basename(dm_path),
                      ", ".join(f"{a} -> {b}" for a, b in renamed.items()))
    except (ValueError, OSError) as e:
        log.warning("Datamine model %s not written: %s", os.path.basename(dm_path), e)
        dm_path = None
        if csv_path is None:
            res.spool.write(path, res.empty_columns)
            csv_path = path
    return model_grade_tonnage(res), csv_path, dm_path

def _model_cols(row: dict, saved: tuple, model_dir: str) -> GradeTonnage:
    gt, csv_path, dm_path = saved
    root = os.path.dirname(model_dir)
    row["MODEL"] = os.path.relpath(csv_path, root) if csv_path else ""
    row["MODEL_DM"] = os.path.relpath(dm_path, root) if dm_path else ""
    return gt

def run_sensitivity(p: DilutionParams, df, values: dict,
                    progress: Optional[Callable[[str], None]] = None,
                    step: Optional[Callable[[int, int], None]] = None,
                    cancel=None, base: Optional[DilutionResult] = None,
                    pool=None, model_dir: Optional[str] = None,
                    model_csv: bool = True) -> dict:
    from .store import as_store

    store = base.store if df is None and base is not None else as_store(df, p)
    base_row = result_row(base) if base is not None else None
    bx = base.params["block_x"] if base is not None else None
    by = base.params["block_y"] if base is not None else None
    jobs = [(k, v) for k, vals in values.items() for v in vals]
    tables = {k: [] for k in values}
    curves = {k: {} for k in values}
    benches = {k: {} for k in values}
    undiluted = None
    if model_dir and base is not None and base.spool is not None:
        undiluted = _spool_benches(base, base.params["cutoff"])
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    for done, (key, v) in enumerate(jobs, 1):
        row = {"PARAMETER": SENS_PARAMS[key], "VALUE": v}
        try:
            q = dataclasses.replace(p, **{key: v})
            if key == "smu_x":
                parent = bx or _parent_size(p, store)[0]
                q = dataclasses.replace(q, step_x=fit_step(v, p.step_x, parent))
            if key == "smu_y" and p.step_y:
                parent = by or _parent_size(p, store)[1]
                q = dataclasses.replace(q, step_y=fit_step(v, p.step_y, parent))
            bp = base.params if base is not None else {}
            row.update(CUTOFF=q.cutoff, SMU_X=q.smu_x,
                       SMU_Y=q.smu_y or bp.get("smu_y"),
                       SMU_Z=q.smu_z or bp.get("smu_z"),
                       STEP_X=q.step_x,
                       STEP_Y=q.step_y or q.smu_y or bp.get("smu_y"), NOTE="")
            same = base is not None and all(
                abs(float(a) - float(b)) < 1e-9 for a, b in (
                    (q.cutoff, base.params["cutoff"]), (q.smu_x, base.params["smu_x"]),
                    (q.step_x, base.params["step_x"]),
                    (q.smu_y or base.params["block_y"], base.params["smu_y"]),
                    (q.step_y or q.smu_y or base.params["block_y"],
                     base.params["step_y"]),
                    (q.smu_z or base.params["block_z"], base.params["smu_z"])))
            path = (os.path.join(model_dir, sensitivity_model_name(p.mode, key, v))
                    if model_dir else None)
            if same and (path is None or base.spool is not None):
                row.update(base_row)
                row["NOTE"] = "base case"
                if path:
                    benches[key][v] = _spool_benches(base, row["CUTOFF"])
                    curves[key][v] = _model_cols(row, _save_model(base, path, model_csv),
                                                 model_dir)
            else:
                res = run_dilution(q, progress=lambda s: None, store=store,
                                   cancel=cancel, summary_only=path is None, pool=pool)
                try:
                    row.update(result_row(res))
                    row["SMU_Y"] = res.params["smu_y"]
                    row["SMU_Z"] = res.params["smu_z"]
                    row["STEP_Y"] = res.params["step_y"]
                    if path:
                        benches[key][v] = _spool_benches(res, q.cutoff)
                        curves[key][v] = _model_cols(row, _save_model(res, path, model_csv),
                                                     model_dir)
                finally:
                    if res.spool is not None:
                        res.spool.close()
                if same:
                    row["NOTE"] = "base case"
        except ValueError as e:
            row["NOTE"] = str(e)
        tables[key].append(row)
        if progress:
            progress(f"Sensitivity {done}/{len(jobs)}: {SENS_PARAMS[key]} {v:g} "
                     + (row["NOTE"] if row["NOTE"] and row["NOTE"] != "base case"
                        else f"-> {row.get('DILUTED_TONNES', 0):,.0f} t "
                             f"@ {row.get('DILUTED_AU', 0):.3f}"))
        if step:
            step(done, len(jobs))
    cols = ["PARAMETER", "VALUE", "CUTOFF", "SMU_X", "SMU_Y", "SMU_Z", "STEP_X",
            "STEP_Y"]
    tail = ["NOTE"] + (["MODEL", "MODEL_DM"] if model_dir else [])
    out = {}
    for k, rows in tables.items():
        t = pd.DataFrame(rows)
        for c in cols + tail:
            if c not in t.columns:
                t[c] = None
        out[k] = t[cols + [c for c in t.columns if c not in cols + tail] + tail]
        out[k].attrs["benches"] = benches[k]
        out[k].attrs["undiluted"] = undiluted
        if model_dir:
            out[k].attrs["curves"] = curves[k]
    return out

def _parent_size(p: DilutionParams, store) -> tuple[float, float, float]:
    from .core import infer_block_size
    return infer_block_size(store, p)

def _saved(row: dict):
    return row.get("MODEL") or row.get("MODEL_DM")

def _read_saved_benches(model_dm, model_csv, root: str, p: DilutionParams,
                        cutoff: float) -> pd.DataFrame:
    from .core import iter_diluted_dm
    if isinstance(model_dm, str) and model_dm and os.path.isfile(os.path.join(root, model_dm)):
        try:
            return model_benches(iter_diluted_dm(os.path.join(root, model_dm), p), p, cutoff)
        except (ValueError, OSError) as e:
            log.warning("Could not read %s (%s); trying the CSV", model_dm, e)
    if isinstance(model_csv, str) and model_csv and os.path.isfile(
            os.path.join(root, model_csv)):
        return model_benches(pd.read_csv(os.path.join(root, model_csv), chunksize=500_000),
                             p, cutoff)
    raise ValueError("no saved model")

def bench_cases(tables: dict, root: Optional[str] = None,
                p: Optional[DilutionParams] = None) -> list[tuple[str, pd.DataFrame]]:
    cases, seen_base = [], False
    for k, t in tables.items():
        for r in t.to_dict("records"):
            note = r.get("NOTE") if isinstance(r.get("NOTE"), str) else ""
            if note not in ("", "base case"):
                continue
            base = note == "base case"
            if base and seen_base:
                continue
            frame = t.attrs.get("benches", {}).get(r["VALUE"])
            if frame is None or not len(frame):
                if root is None or p is None:
                    log.warning("No bench figures for %s %g", SENS_PARAMS[k], r["VALUE"])
                    continue
                try:
                    frame = _read_saved_benches(r.get("MODEL_DM"), r.get("MODEL"),
                                                root, p, float(r["CUTOFF"]))
                except ValueError as e:
                    log.warning("Bench figures for %s %g not read: %s",
                                SENS_PARAMS[k], r["VALUE"], e)
                    continue
            seen_base |= base
            cases.append(("Base case" if base else f"{SENS_PARAMS[k]} {r['VALUE']:g}",
                          frame))
    return cases

def bench_table(cases: list, undiluted: Optional[pd.DataFrame]) -> pd.DataFrame:
    series = []
    if undiluted is not None and len(undiluted):
        series.append(("Undiluted", undiluted, "UND_T", "UND_M"))
    series += [(f"{label} Diluted", f, "DIL_T", "DIL_M") for label, f in cases]
    out = pd.DataFrame({"Bench RL": sorted(
        {z for _, f, _, _ in series for z in f.ZC.tolist()}, reverse=True)})
    for name, f, tc, mc in series:
        f = f.drop_duplicates("ZC").set_index("ZC")
        tonnes, metal = out["Bench RL"].map(f[tc]), out["Bench RL"].map(f[mc])
        out[f"{name} Tonnes"] = tonnes
        out[f"{name} Grade"] = (metal / tonnes.where(tonnes > 0)).fillna(0.0).where(
            tonnes.notna())
        out[f"{name} Ounces"] = metal / GRAMS_PER_OZ
    return out

def write_sensitivity(tables: dict, out_dir: str, mode: str,
                      run_info: Optional[dict] = None) -> dict:
    paths = {}
    for k, t in tables.items():
        path = os.path.join(out_dir, f"sensitivity_{k}_{mode}.csv")
        t.to_csv(path, index=False)
        paths[f"sensitivity_{k}"] = path
    cases = bench_cases(tables)
    if cases:
        undiluted = next((t.attrs.get("undiluted") for t in tables.values()
                          if t.attrs.get("undiluted") is not None), None)
        path = os.path.join(out_dir, f"sensitivity_benches_{mode}.csv")
        bench_table(cases, undiluted).to_csv(path, index=False)
        paths["sensitivity_benches"] = path
    curves = sensitivity_curves(tables)
    if len(curves):
        path = os.path.join(out_dir, f"sensitivity_curves_{mode}.csv")
        curves.to_csv(path, index=False)
        paths["sensitivity_curves"] = path
    return paths

def write_bench_csv_from_run(run_dir: str, p: DilutionParams, mode: Optional[str] = None,
                             out_path: Optional[str] = None) -> str:
    mode = mode or p.mode
    tables = {}
    for k in SENS_PARAMS:
        f = os.path.join(run_dir, f"sensitivity_{k}_{mode}.csv")
        if os.path.isfile(f):
            tables[k] = pd.read_csv(f)
    if not tables:
        raise ValueError(f"No sensitivity tables for {mode} in {run_dir}")
    stem = os.path.splitext(diluted_model_name(mode))[0]
    try:
        undiluted = _read_saved_benches(stem + ".dm", stem + ".csv", run_dir, p, p.cutoff)
    except ValueError:
        log.warning("No main diluted model in %s: no undiluted columns", run_dir)
        undiluted = None
    cases = bench_cases(tables, run_dir, p)
    if not cases:
        raise ValueError(f"No saved sensitivity models to read in {run_dir}")
    out_path = out_path or os.path.join(run_dir, f"sensitivity_benches_{mode}.csv")
    bench_table(cases, undiluted).to_csv(out_path, index=False)
    return out_path

def curve_cutoffs(bins: list, points: int = 100) -> np.ndarray:
    top = max([b.quantile(0.9) for b in bins] + [GT_STEP])
    for d in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0):
        if top / d <= points:
            break
    n = int(math.ceil(top / d - 1e-9))
    return np.round(np.arange(n + 1) * d, 6)

def sensitivity_curves(tables: dict) -> pd.DataFrame:
    runs = [(k, v, gt) for k, t in tables.items()
            for v, gt in t.attrs.get("curves", {}).items()]
    if not runs:
        return pd.DataFrame()
    cutoffs = curve_cutoffs([gt for _, _, gt in runs])
    parts = []
    for k, v, gt in runs:
        t = tables[k]
        base = bool((t.loc[t.VALUE == v, "NOTE"] == "base case").any())
        c = gt.curve(cutoffs)
        c.insert(0, "BASE", base)
        c.insert(0, "VALUE", v)
        c.insert(0, "PARAMETER", SENS_PARAMS[k])
        parts.append(c)
    return pd.concat(parts, ignore_index=True)

def format_sensitivity(tables: dict) -> str:
    lines = []
    for k, t in tables.items():
        lines.append(f"{SENS_PARAMS[k]} sensitivity")
        lines.append(f"{'Value':>8}{'X step':>7}{'Y step':>7}{'Diluted t':>14}"
                     f"{'AU':>8}{'Oz':>11}{'Dil %':>8}{'Loss %':>8}")
        for r in t.to_dict("records"):
            if r["NOTE"] and r["NOTE"] != "base case":
                lines.append(f"{r['VALUE']:>8g}{'':>14}  {r['NOTE']}")
            else:
                lines.append(f"{r['VALUE']:>8g}{r['STEP_X']:>7g}{r['STEP_Y']:>7g}"
                             f"{r['DILUTED_TONNES']:>14,.0f}{r['DILUTED_AU']:>8.3f}"
                             f"{r['DILUTED_OZ']:>11,.0f}{r['DILUTION_PCT']:>8.1f}"
                             f"{r['ORE_LOSS_PCT']:>8.1f}"
                             + ("  base case" if r["NOTE"] == "base case" else ""))
        lines.append("")
    return "\n".join(lines)

def append_history(res: DilutionResult, seconds: float,
                   history_file: str = HISTORY_FILE) -> dict:
    p = res.params
    try:
        st = os.stat(p["input_path"])
        size, mtime = st.st_size, _dt.datetime.fromtimestamp(st.st_mtime).isoformat(
            timespec="seconds")
    except OSError:
        size, mtime = None, None
    row = {
        "RUN_TIME": _dt.datetime.now().isoformat(timespec="seconds"),
        "USER": getpass.getuser(),
        "INPUT": p["input_path"], "INPUT_BYTES": size, "INPUT_MODIFIED": mtime,
        "OUTPUT": p["output_dir"], "MODE": p["mode"], "CUTOFF": p["cutoff"],
        "SMU_X": p["smu_x"], "SMU_Y": p["smu_y"], "SMU_Z": p["smu_z"],
        "STEP_X": p["step_x"], "STEP_Y": p.get("step_y"),
        "PARENT": f"{p['block_x']} x {p['block_y']} x {p['block_z']}",
        "ORIGIN": f"{p['origin_x']}, {p['origin_y']}, {p['origin_z']}",
        "THREADS": p.get("threads"), "SECONDS": round(seconds, 1),
        **result_row(res),
    }
    for path in (history_file, os.path.join(p["output_dir"], "run_history.csv")):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _append_row(path, row)
        except OSError as e:
            log.warning("Could not update the run history %s: %s", path, e)
    return row

def _append_row(path: str, row: dict) -> None:
    frame = pd.DataFrame([row])
    if not os.path.exists(path):
        frame.to_csv(path, index=False)
        return
    header = list(pd.read_csv(path, nrows=0).columns)
    if header == list(frame.columns):
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        pd.concat([pd.read_csv(path), frame], ignore_index=True).to_csv(path, index=False)

def read_history(history_file: str = HISTORY_FILE) -> pd.DataFrame:
    if not os.path.exists(history_file):
        return pd.DataFrame()
    return pd.read_csv(history_file)
