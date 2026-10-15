import os
import sys

import dataclasses

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dilution.core import GRAMS_PER_OZ, run_dilution, write_outputs
from dilution.extras import (run_sensitivity, write_bench_csv_from_run,
                             write_sensitivity)
from test_core import _params

VALUES = {"cutoff": [0.5, 1.0, 2.0], "smu_x": [5.0, 10.0, 20.0]}
BASE = "Base case"

def _benched_model():
    rng = np.random.default_rng(5)
    return pd.DataFrame([
        dict(XC=2.5 + 5 * i, YC=2.5 + 5 * j, ZC=2.5 + 5 * k,
             AU=float(rng.lognormal(-0.5, 1.0)), RESCAT=2, DENSITY=2.0)
        for i in range(24) for j in range(4) for k in range(4)])

def _run(tmp_path, dm_only=False, audit=False):
    p = _params(smu_x=10.0, step_x=5.0, audit_fields=audit, output_dir=str(tmp_path))
    df = _benched_model()
    base = run_dilution(p, df=df)
    write_outputs(base, str(tmp_path), diluted_csv=not dm_only)
    t = run_sensitivity(p, df, VALUES, base=base,
                        model_dir=str(tmp_path / "sensitivity_models"),
                        model_csv=not dm_only)
    paths = write_sensitivity(t, str(tmp_path), "MII", base.params)
    return p, base, t, paths

def _above(model, grade, cutoff):
    m = model[model[grade] >= cutoff].copy()
    m["T"] = m.XINC * m.YINC * m.ZINC * m.DENSITY
    m["M"] = m["T"] * m[grade]
    g = m.groupby("ZC")[["T", "M"]].sum()
    return g["T"], g["M"] / g["T"], g["M"] / GRAMS_PER_OZ

def test_columns_and_rows(tmp_path):
    p, base, t, paths = _run(tmp_path)
    b = pd.read_csv(paths["sensitivity_benches"])
    cases = ["Cut-off 0.5", BASE, "Cut-off 2", "SMU X 5", "SMU X 20"]
    assert list(b.columns) == ["Bench RL"] + [
        f"{c} {m}" for c in ["Undiluted"] + [f"{x} Diluted" for x in cases]
        for m in ("Tonnes", "Grade", "Ounces")]
    assert b["Bench RL"].tolist() == sorted(base.by_bench.ZC, reverse=True)
    assert b["Bench RL"].is_unique and not b.isna().any().any()

    def check(prefix, model, grade, cutoff):
        tn, gr, oz = _above(model, grade, cutoff)
        z = b["Bench RL"]
        assert b[f"{prefix} Tonnes"].tolist() == pytest.approx(
            tn.reindex(z).fillna(0).tolist())
        assert b[f"{prefix} Grade"].tolist() == pytest.approx(
            gr.reindex(z).fillna(0).tolist())
        assert b[f"{prefix} Ounces"].tolist() == pytest.approx(
            oz.reindex(z).fillna(0).tolist())

    check("Undiluted", base.diluted_model, "AU", p.cutoff)

    for case, key, v in [("Cut-off 0.5", "cutoff", 0.5), (BASE, "cutoff", 1.0),
                         ("Cut-off 2", "cutoff", 2.0), ("SMU X 5", "smu_x", 5.0),
                         ("SMU X 20", "smu_x", 20.0)]:
        r = dataclasses.replace(p, **{key: v})
        if key == "smu_x":
            r = dataclasses.replace(r, step_x=min(v, 5.0))
        check(f"{case} Diluted", run_dilution(r, df=_benched_model()).diluted_model,
              "D_AU", r.cutoff)

    ore = base.diluted_model.query("MATERIAL == 'ORE'")
    assert b["Undiluted Tonnes"].sum() > _above(ore, "AU", p.cutoff)[0].sum()

def test_audit_model_uses_the_definition_grade_for_undiluted(tmp_path):
    p, base, _, paths = _run(tmp_path, audit=True)
    b = pd.read_csv(paths["sensitivity_benches"])
    tn, _, _ = _above(base.diluted_model, "AU_MII", p.cutoff)
    assert b["Undiluted Tonnes"].tolist() == pytest.approx(
        tn.reindex(b["Bench RL"]).fillna(0).tolist())

@pytest.mark.parametrize("dm_only", [False, True])
def test_made_from_a_loaded_run(tmp_path, dm_only):
    p, _, _, paths = _run(tmp_path, dm_only)
    live = pd.read_csv(paths["sensitivity_benches"])
    out = write_bench_csv_from_run(str(tmp_path), p, "MII",
                                   str(tmp_path / "loaded.csv"))
    pd.testing.assert_frame_equal(pd.read_csv(out), live)

def test_csv_used_when_the_dm_is_missing(tmp_path):
    p, _, _, paths = _run(tmp_path)
    live = pd.read_csv(paths["sensitivity_benches"])
    for f in (tmp_path / "sensitivity_models").glob("*.dm"):
        f.unlink()
    out = write_bench_csv_from_run(str(tmp_path), p, "MII", str(tmp_path / "l.csv"))
    pd.testing.assert_frame_equal(pd.read_csv(out), live)

def test_smu_z_case_with_other_benches_leaves_blanks(tmp_path):
    p = _params(smu_x=10.0, step_x=5.0, audit_fields=False)
    df = _benched_model()
    base = run_dilution(p, df=df)
    t = run_sensitivity(p, df, {"smu_z": [10.0]}, base=base,
                        model_dir=str(tmp_path / "sensitivity_models"))
    paths = write_sensitivity(t, str(tmp_path), "MII", base.params)
    b = pd.read_csv(paths["sensitivity_benches"])
    assert "SMU Z 10 Diluted Tonnes" in b.columns
    assert b["SMU Z 10 Diluted Tonnes"].notna().sum() >= 1

def test_no_bench_csv_without_runs(tmp_path):
    p = _params(smu_x=10.0, step_x=5.0)
    paths = write_sensitivity({}, str(tmp_path), "MII")
    assert paths == {}
