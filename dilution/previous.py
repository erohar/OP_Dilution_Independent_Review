from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .core import (CHUNK_ROWS, DilutionParams, diluted_model_name,
                   iter_diluted_dm)
from .extras import (SENS_PARAMS, GradeTonnage, _ore_cells, sensitivity_curves,
                     write_bench_csv_from_run)
from .regions import as_regions

log = logging.getLogger(__name__)

PARAMS_FILE = "run_parameters.json"

def read_settings(path: str) -> dict:
    try:
        with open(path) as f:
            settings = json.load(f)
    except OSError as e:
        raise ValueError(f"Could not read the settings {path}: {e.strerror}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"The settings file {path} is not valid JSON: {e}") from e
    if not isinstance(settings, dict):
        raise ValueError(f"{path} is not a settings file")
    return settings

def output_folder(settings: dict, settings_path: Optional[str] = None) -> str:
    out = settings.get("output") or ""
    if out and os.path.isdir(out):
        return out
    here = os.path.dirname(os.path.abspath(settings_path)) if settings_path else ""
    if here and os.path.isfile(os.path.join(here, PARAMS_FILE)):
        return here
    raise ValueError(f"The output folder of the run was not found: {out or '(none saved)'}")

def load_params(folder: str) -> tuple[dict, DilutionParams]:
    path = os.path.join(folder, PARAMS_FILE)
    if not os.path.isfile(path):
        raise ValueError(f"No {PARAMS_FILE} in {folder}: nothing to load")
    with open(path) as f:
        saved = json.load(f)
    names = {fl.name for fl in dataclasses.fields(DilutionParams)}
    kw = {k: v for k, v in saved.items() if k in names}
    kw["regions"] = as_regions(saved.get("regions") or [])
    return saved, DilutionParams(**kw)

def model_path(folder: str, mode: str) -> Optional[str]:
    csv = os.path.join(folder, diluted_model_name(mode))
    dm = os.path.splitext(csv)[0] + ".dm"
    return next((p for p in (dm, csv) if os.path.isfile(p)), None)

def _short(name: str, columns) -> str:
    return name if name in columns else name.upper()[:8]

def iter_model(path: str, p: DilutionParams, rows: int = CHUNK_ROWS
               ) -> Iterator[pd.DataFrame]:
    if path.lower().endswith(".dm"):
        yield from iter_diluted_dm(path, p, rows=rows)
    else:
        yield from pd.read_csv(path, chunksize=rows)

def model_grade_curve(path: str, p: DilutionParams) -> GradeTonnage:
    gt = GradeTonnage()
    for ch in iter_model(path, p):
        ch, t = _ore_cells(ch, p)
        gt.add(ch[_short(f"D_{p.grade_col}", ch.columns)].to_numpy(), t)
    return gt

class _OreBlocks:

    def __init__(self, path: str):
        self.path = path
        self.oxides: set = set()
        if os.path.isfile(path):
            head = pd.read_csv(path, nrows=0).columns
            if "OXIDE" in head:
                for ch in pd.read_csv(path, usecols=["OXIDE"], chunksize=CHUNK_ROWS):
                    self.oxides.update(ch["OXIDE"].astype(str).unique())

    def bench(self, zc: float) -> pd.DataFrame:
        if not os.path.isfile(self.path):
            return pd.DataFrame()
        parts = [ch[(ch["ZC"] - zc).abs() < 1e-6]
                 for ch in pd.read_csv(self.path, chunksize=CHUNK_ROWS)]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

class LoadedRun:

    def __init__(self, folder: str):
        self.folder = folder
        self.params, self.run_params = load_params(folder)
        p = self.params
        self.summary = pd.read_csv(os.path.join(folder, "summary.csv"))
        self.by_bench = pd.read_csv(os.path.join(folder, "summary_by_bench.csv"))
        self.ore_spool = _OreBlocks(os.path.join(folder, "ore_blocks.csv"))
        self.model = model_path(folder, p["mode"])
        if self.model is None:
            raise ValueError(f"No diluted model (.dm or CSV) in {folder}")
        try:
            self.grid = dict(
                xlo=p["origin_x"], xhi=p["origin_x"] + p["parent_nx"] * p["block_x"],
                ylo=p["origin_y"], yhi=p["origin_y"] + p["parent_ny"] * p["block_y"])
        except KeyError as e:
            raise ValueError(f"{PARAMS_FILE} has no {e.args[0]}") from e
        self._last: Optional[tuple] = None

    def close(self) -> None:
        pass

    def bench_cells(self, zc: float) -> dict:
        if self._last and self._last[0] == zc:
            return self._last[1]
        p = self.run_params
        half = self.params["smu_z"] / 2
        parts = []
        for ch in iter_model(self.model, p):
            parts.append(ch[(ch[p.z_col if p.z_col in ch else "ZC"] - zc).abs()
                            < half - 1e-6])
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        cols = {k: (v if v in df.columns else d) for k, v, d in (
            ("X", p.x_col, "XC"), ("Y", p.y_col, "YC"), ("Z", p.z_col, "ZC"),
            ("xs", p.xinc_col or "XINC", "XINC"), ("ys", p.yinc_col or "YINC", "YINC"))}
        grade = _short(f"D_{p.grade_col}", df.columns)
        if not len(df):
            out = {k: np.zeros(0) for k in ("X", "Y", "Z", "xs", "ys", "def", "ore")}
        else:
            out = {k: df[c].to_numpy(float) for k, c in cols.items()}
            out["def"] = df[grade].to_numpy(float)
            out["ore"] = (df["MATERIAL"].astype(str) == "ORE").to_numpy(float)
        self._last = (zc, out)
        return out

def load_run(settings_path: str) -> LoadedRun:
    settings = read_settings(settings_path)
    return LoadedRun(output_folder(settings, settings_path))

def rebuild_sensitivity(folder: str, out_dir: Optional[str] = None) -> dict:
    saved, p = load_params(folder)
    mode = saved["mode"]
    out = out_dir or folder
    os.makedirs(out, exist_ok=True)
    tables = {}
    for key in SENS_PARAMS:
        path = os.path.join(folder, f"sensitivity_{key}_{mode}.csv")
        if not os.path.isfile(path):
            continue
        t = pd.read_csv(path, float_precision="round_trip")
        for c in ("NOTE", "MODEL", "MODEL_DM"):
            if c in t.columns:
                t[c] = t[c].fillna("").astype(str)
        curves = {}
        for r in t.to_dict("records"):
            rel = r.get("MODEL_DM") or r.get("MODEL")
            if not rel:
                continue
            model = os.path.join(folder, rel)
            if not os.path.isfile(model):
                log.warning("Sensitivity model %s not found; left out of the curves", model)
                continue
            curves[r["VALUE"]] = model_grade_curve(model, p)
        if curves:
            t.attrs["curves"] = curves
        tables[key] = t
    if not tables:
        raise ValueError(f"No sensitivity tables for mode {mode} in {folder}")
    paths = {}
    curve_table = sensitivity_curves(tables)
    if len(curve_table):
        paths["sensitivity_curves"] = os.path.join(out, f"sensitivity_curves_{mode}.csv")
        curve_table.to_csv(paths["sensitivity_curves"], index=False)
    paths["sensitivity_benches"] = write_bench_csv_from_run(
        folder, p, mode, os.path.join(out, f"sensitivity_benches_{mode}.csv"))
    return paths
