import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dilution.core import (DilutionParams, run_dilution, GRAMS_PER_OZ,
                           format_report, write_outputs)
from dilution.checks import check_outputs
from dilution.extras import result_row

def _strip_model(grades, rescat=None, bx=5.0, oxide=None, ny=1):
    n = len(grades)
    rows = []
    for iy in range(ny):
        df = pd.DataFrame({
            "XC": np.arange(n) * bx + bx / 2,
            "YC": bx / 2 + iy * bx,
            "ZC": 2.5,
            "AU": grades,
            "RESCAT": rescat if rescat is not None else [2] * n,
            "DENSITY": 2.0,
        })
        if oxide is not None:
            df["OXIDE"] = oxide
        rows.append(df)
    return pd.concat(rows, ignore_index=True)

def _params(**kw):
    base = dict(input_path="mem", output_dir="out", block_x=5, block_y=5,
                block_z=5, cutoff=1.0, smu_x=10.0, step_x=5.0, oxide_col=None,
                ore_rule="value", audit_fields=True)
    base.update(kw)
    return DilutionParams(**base)

def test_classification_columns_in_by_bench():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    res = run_dilution(_params(), df=df)
    bb = res.by_bench
    for col in ("ORE_ORE_TONNES", "ORE_ORE_METAL",
                "ORE_WASTE_TONNES", "ORE_WASTE_METAL", "ORE_WASTE_OZ",
                "WASTE_ORE_TONNES", "WASTE_ORE_METAL",
                "WASTE_WASTE_TONNES", "WASTE_WASTE_METAL"):
        assert col in bb.columns, f"Missing column {col}"

def test_classification_in_summary():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    res = run_dilution(_params(), df=df)
    items = res.summary.ITEM.tolist()
    assert any("Ore-Ore" in i for i in items)
    assert any("Ore-Waste" in i for i in items)
    assert any("Waste-Ore" in i for i in items)
    assert any("Waste-Waste" in i for i in items)
    assert "ore_loss_oz" in res.summary.attrs
    assert "dilution_tonnes" in res.summary.attrs

def test_classification_sums_to_undiluted_and_diluted():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    res = run_dilution(_params(), df=df)
    bb = res.by_bench
    oo = bb.ORE_ORE_TONNES.sum()
    ow = bb.ORE_WASTE_TONNES.sum()
    wo = bb.WASTE_ORE_TONNES.sum()
    und = bb.UNDILUTED_TONNES.sum()
    dil = bb.DILUTED_TONNES.sum()
    assert oo + ow == pytest.approx(und, abs=1e-6)
    assert oo + wo == pytest.approx(dil, abs=1e-6)

def test_single_high_block_has_ore_ore_and_waste_ore():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    res = run_dilution(_params(), df=df)
    bb = res.by_bench

    assert bb.ORE_ORE_TONNES.sum() > 0

    assert bb.WASTE_ORE_TONNES.sum() > 0

    assert bb.ORE_WASTE_TONNES.sum() == pytest.approx(0, abs=1e-9)

def test_all_above_cutoff_no_ore_waste():
    df = _strip_model([3, 3, 3, 3])
    res = run_dilution(_params(), df=df)
    bb = res.by_bench
    assert bb.ORE_WASTE_TONNES.sum() == pytest.approx(0, abs=1e-9)
    assert bb.ORE_ORE_TONNES.sum() == pytest.approx(bb.UNDILUTED_TONNES.sum(), abs=1e-6)

def test_all_below_cutoff_no_ore_ore():
    df = _strip_model([0.1, 0.1, 0.1, 0.1])
    res = run_dilution(_params(), df=df)
    bb = res.by_bench
    assert bb.ORE_ORE_TONNES.sum() == pytest.approx(0, abs=1e-9)
    assert bb.ORE_WASTE_TONNES.sum() == pytest.approx(0, abs=1e-9)
    assert bb.WASTE_ORE_TONNES.sum() == pytest.approx(0, abs=1e-9)

def test_classification_metal_consistency():
    df = _strip_model([0, 0, 3, 3, 0, 3, 3, 0, 0])
    res = run_dilution(_params(), df=df)
    s = res.summary
    und_row = s.iloc[0]
    dil_row = s.iloc[1]
    bb = res.by_bench
    oo_m = bb.ORE_ORE_METAL.sum()
    ow_m = bb.ORE_WASTE_METAL.sum()
    und_m = und_row.METAL_OZ * GRAMS_PER_OZ
    assert oo_m + ow_m == pytest.approx(und_m, abs=1e-6)

    wo_m = bb.WASTE_ORE_METAL.sum()
    dil_m = dil_row.METAL_OZ * GRAMS_PER_OZ
    assert oo_m + wo_m == pytest.approx(dil_m, abs=1e-6)

def test_ore_loss_oz_in_summary():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    res = run_dilution(_params(), df=df)
    s = res.summary
    ow_metal = res.by_bench.ORE_WASTE_METAL.sum()
    assert s.attrs["ore_loss_oz"] == pytest.approx(ow_metal / GRAMS_PER_OZ, abs=1e-9)

def test_dilution_tonnes_in_summary():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    res = run_dilution(_params(), df=df)
    s = res.summary
    wo_t = res.by_bench.WASTE_ORE_TONNES.sum()
    assert s.attrs["dilution_tonnes"] == pytest.approx(wo_t, abs=1e-9)

def test_classification_in_report():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    res = run_dilution(_params(), df=df)
    report = format_report(res)
    assert "Ore loss (ounces)" in report
    assert "Dilution (tonnes)" in report

def test_classification_in_result_row():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    res = run_dilution(_params(), df=df)
    row = result_row(res)
    assert "ORE_LOSS_OZ" in row
    assert "DILUTION_TONNES" in row
    assert row["ORE_LOSS_OZ"] == pytest.approx(res.summary.attrs["ore_loss_oz"], abs=1e-9)
    assert row["DILUTION_TONNES"] == pytest.approx(res.summary.attrs["dilution_tonnes"], abs=1e-9)

def test_check_outputs_classification():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    p = _params()
    res = run_dilution(p, df=df)
    checks = check_outputs(df, p, res)
    names = [c.name for c in checks]
    assert "Classification sums to undiluted" in names
    assert "Classification sums to diluted" in names
    for c in checks:
        if "Classification" in c.name:
            assert c.status == "PASS", f"{c.name}: {c.detail}"

def test_ore_waste_oz_in_by_bench():
    df = _strip_model([0, 0, 3, 3, 0, 3, 3, 0, 0])
    res = run_dilution(_params(), df=df)
    bb = res.by_bench
    expected = bb.ORE_WASTE_METAL / GRAMS_PER_OZ
    np.testing.assert_allclose(bb.ORE_WASTE_OZ, expected, atol=1e-9)

def test_classification_with_mixed_grades():

    df = _strip_model([0, 3, 0, 0.5, 3, 0, 0, 0])
    res = run_dilution(_params(), df=df)
    bb = res.by_bench

    assert bb.ORE_ORE_TONNES.sum() > 0

    assert bb.WASTE_ORE_TONNES.sum() > 0

    assert bb.WASTE_WASTE_TONNES.sum() > 0

    assert bb.ORE_ORE_TONNES.sum() + bb.ORE_WASTE_TONNES.sum() == pytest.approx(
        bb.UNDILUTED_TONNES.sum(), abs=1e-6)
    assert bb.ORE_ORE_TONNES.sum() + bb.WASTE_ORE_TONNES.sum() == pytest.approx(
        bb.DILUTED_TONNES.sum(), abs=1e-6)
