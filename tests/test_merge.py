import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dilution.core import DilutionParams, run_dilution
from dilution.merge import MergeSpec, merge_cells

def _model(split_alike=True):
    rng = np.random.default_rng(3)
    rows = []
    for i in range(3):
        for j in range(3):
            for k in range(2):
                au = round(float(rng.lognormal(0.0, 0.7)), 3)
                rc = int(rng.integers(1, 4))
                dens = round(float(rng.uniform(2.2, 2.9)), 3)
                for qx in range(2):
                    for qy in range(2):
                        odd = (i + j + k) % 3 == 0 and qx == 1 and qy == 1
                        a = au if split_alike and not odd else round(au + qx + 2 * qy + 0.1, 3)
                        rows.append(dict(
                            XC=i * 20 + 5 + qx * 10, YC=j * 20 + 5 + qy * 10,
                            ZC=k * 10 + 5, XINC=10.0, YINC=10.0, ZINC=10.0,
                            AU=a, RESCAT=rc, DENSITY=dens))
    return pd.DataFrame(rows)

def _params(**kw):
    base = dict(input_path="mem", output_dir="out", cutoff=1.0, smu_x=10.0,
                smu_y=10.0, smu_z=10.0, step_x=1.0, step_y=1.0, oxide_col=None,
                xinc_col="XINC", yinc_col="YINC", zinc_col="ZINC",
                block_x=20.0, block_y=20.0, block_z=10.0,
                origin_x=0.0, origin_y=0.0, origin_z=0.0)
    base.update(kw)
    return DilutionParams(**base)

def _tm(dm, au):
    t = dm.XINC * dm.YINC * dm.ZINC * dm.DENSITY
    return t.sum(), (t * dm[au]).sum()

@pytest.mark.parametrize("audit", [False, True])
def test_merge_keeps_tonnes_and_metal_and_reduces_cells(audit):
    df = _model()
    off = run_dilution(_params(audit_fields=audit), df=df)
    on = run_dilution(_params(audit_fields=audit, merge_subcells=True), df=df)
    a, b = off.diluted_model, on.diluted_model
    assert len(b) < len(a)
    assert on.stats["merged_subcells"] == len(a) - len(b)
    assert on.stats["diluted_cells"] == len(b)
    ta, ma = _tm(a, "D_AU")
    tb, mb = _tm(b, "D_AU")
    assert tb == pytest.approx(ta, rel=1e-9)
    assert mb == pytest.approx(ma, rel=1e-9)
    if audit:
        assert b.TONNES.sum() == pytest.approx(a.TONNES.sum(), rel=1e-9)
        assert b.METAL_G.sum() == pytest.approx(a.METAL_G.sum(), rel=1e-9)

def test_off_is_unchanged():
    df = _model()
    default = run_dilution(_params(), df=df).diluted_model
    off = run_dilution(_params(merge_subcells=False), df=df).diluted_model
    pd.testing.assert_frame_equal(default, off)
    assert run_dilution(_params(), df=df).stats["merged_subcells"] == 0

def test_nothing_mergeable_is_unchanged():
    spec = MergeSpec(origin=(0, 0, 0), parent=(10, 10, 10))
    cells = pd.DataFrame(dict(XC=[2.5, 7.5, 2.5], YC=[5.0, 5.0, 5.0],
                              ZC=[2.5, 2.5, 7.5], XINC=[5.0] * 3, YINC=[10.0] * 3,
                              ZINC=[5.0] * 3, AU=[1.0, 2.0, 3.0]))
    pd.testing.assert_frame_equal(merge_cells(cells, spec), cells)

def test_cells_stay_in_one_parent_and_differing_cells_do_not_merge():
    df = _model()
    on = run_dilution(_params(merge_subcells=True), df=df).diluted_model
    for c, w, size in (("XC", "XINC", 20.0), ("YC", "YINC", 20.0), ("ZC", "ZINC", 10.0)):
        lo = np.floor((on[c] - on[w] / 2) / size + 1e-9)
        hi = np.floor((on[c] + on[w] / 2) / size - 1e-9)
        assert (lo == hi).all()

    spec = MergeSpec(origin=(0, 0, 0), parent=(20, 20, 10),
                     density="DENSITY")
    again = merge_cells(on, spec)
    assert len(again) == len(on)

def test_merge_cells_unit():
    spec = MergeSpec(origin=(0, 0, 0), parent=(10, 10, 10), density="DENSITY",
                     additive=["TONNES", "VOLUME"])
    cells = pd.DataFrame(dict(
        XC=[2.5, 7.5, 2.5, 7.5], YC=[5.0, 5.0, 5.0, 5.0], ZC=[2.5, 2.5, 7.5, 7.5],
        XINC=[5.0] * 4, YINC=[10.0] * 4, ZINC=[5.0] * 4,
        AU=[1.0, 1.0, 1.0, 2.0], DENSITY=[2.0] * 4,
        TONNES=[500.0] * 4, VOLUME=[250.0] * 4))
    out = merge_cells(cells, spec)
    assert len(out) == 3
    top = out[out.ZC == 2.5].iloc[0]
    assert (top.XC, top.XINC, top.TONNES, top.VOLUME) == (5.0, 10.0, 1000.0, 500.0)
    assert top.DENSITY == pytest.approx(2.0)

def test_merge_across_parents_never():
    spec = MergeSpec(origin=(0, 0, 0), parent=(10, 10, 10))
    cells = pd.DataFrame(dict(XC=[5.0, 15.0], YC=[5.0, 5.0], ZC=[5.0, 5.0],
                              XINC=[10.0, 10.0], YINC=[10.0, 10.0],
                              ZINC=[10.0, 10.0], AU=[1.0, 1.0]))
    assert len(merge_cells(cells, spec)) == 2

def test_dm_output_written_with_merge(tmp_path):
    from dilution.core import write_outputs
    res = run_dilution(_params(merge_subcells=True), df=_model())
    paths = write_outputs(res, str(tmp_path))
    assert os.path.exists(paths["diluted_model_dm"])
    assert len(pd.read_csv(paths["diluted_model"])) == len(res.diluted_model)

def _sub_cells():
    return pd.DataFrame(dict(
        XC=[2.5, 7.5], YC=[5.0, 5.0], ZC=[5.0, 5.0],
        XINC=[5.0, 5.0], YINC=[10.0, 10.0], ZINC=[10.0, 10.0],
        AU=[1.0, 4.0], RESCAT=[1, 2], DENSITY=[2.0, 3.0], D_AU=[2.0, 2.0],
        D_RESCAT=[1, 1], MATERIAL=["ORE", "ORE"]))

def _spec(**rules):
    return MergeSpec(origin=(0, 0, 0), parent=(10, 10, 10), density="DENSITY",
                     initial={"AU": "num", "RESCAT": "cat"}, grade="AU", rules=rules)

def test_initial_attributes_group_by_default():
    assert len(merge_cells(_sub_cells(), _spec())) == 2

def test_diluted_attributes_always_group():
    cells = _sub_cells()
    cells.loc[1, "D_AU"] = 2.5
    assert len(merge_cells(cells, _spec(AU="avg_mass", RESCAT="dom_mass"))) == 2

@pytest.mark.parametrize("rule,expected", [
    ("avg_mass", (1.0 * 1000 + 4.0 * 1500) / 2500),
    ("avg_volume", 2.5),
    ("avg_metal", (1.0 * 1000 * 1.0 + 4.0 * 1500 * 4.0) / (1000 * 1.0 + 1500 * 4.0)),
    ("dom_mass", 4.0), ("dom_metal", 4.0)])
def test_combine_rules_for_a_grade(rule, expected):
    out = merge_cells(_sub_cells(), _spec(AU=rule, RESCAT="dom_mass"))
    assert len(out) == 1
    assert out.AU.iloc[0] == pytest.approx(expected)

def test_dominant_volume_picks_larger_volume():
    cells = _sub_cells()
    cells.loc[0, ["XC", "XINC"]] = [3.0, 6.0]
    cells.loc[1, ["XC", "XINC"]] = [8.0, 4.0]
    out = merge_cells(cells, _spec(AU="avg_mass", RESCAT="dom_volume"))
    assert out.RESCAT.iloc[0] == 1
    cells.loc[1, "DENSITY"] = 9.0
    assert merge_cells(cells, _spec(AU="avg_mass", RESCAT="dom_mass")).RESCAT.iloc[0] == 2

def test_supporting_fields_follow_the_merge():
    out = merge_cells(_sub_cells(), _spec(AU="avg_mass", RESCAT="dom_mass")).iloc[0]
    assert (out.XC, out.XINC) == (5.0, 10.0)
    tonnes = 500 * 2.0 + 500 * 3.0
    assert out.XINC * out.YINC * out.ZINC * out.DENSITY == pytest.approx(tonnes)
    assert out.DENSITY == pytest.approx(tonnes / 1000)

def test_categorical_cannot_average():
    from dilution.merge import merge_spec
    p = _params(merge_rules={"RESCAT": "avg_mass"})
    with pytest.raises(ValueError, match="code"):
        merge_spec(["XC", "YC", "ZC", "RESCAT"], p, (0, 0, 0), (20, 20, 10),
                   {"RESCAT": "cat"})

def test_bad_rule_rejected():
    with pytest.raises(ValueError, match="Merge rule"):
        _params(merge_rules={"AU": "median"}).validate()

def test_run_with_rules_merges_more_and_keeps_tonnes():
    df = _model(split_alike=False)
    big = dict(smu_x=20.0, smu_y=20.0)
    grouped = run_dilution(_params(merge_subcells=True, **big), df=df)
    combined = run_dilution(_params(merge_subcells=True, **big,
                                    merge_rules={"AU": "avg_mass", "RESCAT": "dom_mass"}),
                            df=df)
    off = run_dilution(_params(**big), df=df).diluted_model
    on = combined.diluted_model
    assert combined.stats["merged_subcells"] > grouped.stats["merged_subcells"]
    ta, ma = _tm(off, "D_AU")
    tb, mb = _tm(on, "D_AU")
    assert tb == pytest.approx(ta, rel=1e-9) and mb == pytest.approx(ma, rel=1e-9)

    in_a = (off.XINC * off.YINC * off.ZINC * off.DENSITY * off.AU).sum()
    in_b = (on.XINC * on.YINC * on.ZINC * on.DENSITY * on.AU).sum()
    assert in_b == pytest.approx(in_a, rel=1e-9)

def test_cli_parses_merge_combine():
    import run_dilution as cli
    assert cli._rules("AU=avg_mass, RESCAT=dom_metal") == {"AU": "avg_mass",
                                                          "RESCAT": "dom_metal"}
    with pytest.raises(ValueError):
        cli._rules("AU")
