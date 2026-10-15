import itertools
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dilution.core import (DilutionParams, breakdown_zone,
                           check_smu_ratio, optimise_segments, refine_zone,
                           run_dilution, write_outputs)

def _runs(lab):
    d = np.diff(np.concatenate([[-1], lab, [-1]]))
    idx = np.flatnonzero(d != 0)
    return [(lab[a], b - a) for a, b in zip(idx[:-1], idx[1:])]

def _feasible(lab, m, rs, re):
    runs = _runs(np.asarray(lab))
    for i, (kind, w) in enumerate(runs):
        edge_ok = (i == 0 and rs) or (i == len(runs) - 1 and re)
        if w < m and not (kind == 0 and edge_ok):
            return False
    return True

def _brute(v, m, rs, re):
    best = None
    for lab in itertools.product([0, 1], repeat=len(v)):
        if _feasible(lab, m, rs, re):
            val = float(np.dot(lab, v))
            if best is None or val > best + 1e-12:
                best = val
    return best

@pytest.mark.parametrize("rs,re", [(False, False), (True, True), (True, False),
                                   (False, True)])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_optimiser_matches_brute_force(m, rs, re):
    rng = np.random.default_rng(m * 10 + rs * 2 + re)
    runs = [rng.normal(-0.3, 1.0, size=rng.integers(1, 12)) for _ in range(120)]
    labels = optimise_segments(runs, m, rs, re)
    for v, lab in zip(runs, labels):
        assert len(lab) == len(v)
        want = _brute(v, m, rs, re)
        if want is None:
            assert not lab.any()
            continue
        assert _feasible(lab, m, rs, re), (v, lab)
        assert float(np.dot(lab, v)) == pytest.approx(want, abs=1e-9)

def test_breakdown():
    assert breakdown_zone(4, 4) == [4]
    assert breakdown_zone(7, 4) == [7]
    assert breakdown_zone(8, 4) == [4, 4]
    assert breakdown_zone(13, 4) == [4, 4, 5]

def _P(values):
    return np.concatenate([[0.0], np.cumsum(values)])

def test_refine_keeps_blocks_above_cutoff():
    assert refine_zone(_P([1, 1, 1, 1, 1]), 0, 5, 2, False, False) == [(0, 2), (2, 5)]

def test_refine_returns_low_x_ore_to_previous_block():

    v = [1, 1, 0.5, -2, 1, -3, 5, 5]
    assert refine_zone(_P(v), 0, 8, 2, False, False) == [(0, 3), (6, 8)]

def test_refine_shifts_waste_into_next_smu():

    v = [1, 1, 0.5, -5, -0.2, 1, 1]
    assert refine_zone(_P(v), 0, 7, 2, False, False) == [(0, 3), (5, 7)]

def test_refine_no_shift_when_it_loses_metal():
    v = [1, 1, -0.5, -0.8, 1, -3, 5, 5]
    assert refine_zone(_P(v), 0, 8, 2, False, False) == [(0, 2), (6, 8)]

def _strip_model(grades, rescat=None, bx=5.0, oxide=None, skip=()):
    n = len(grades)
    df = pd.DataFrame({
        "XC": np.arange(n) * bx + bx / 2, "YC": 2.5, "ZC": 2.5,
        "AU": grades, "RESCAT": rescat if rescat is not None else [2] * n,
        "DENSITY": 2.0,
    })
    if oxide is not None:
        df["OXIDE"] = oxide
    return df.drop(index=list(skip)).reset_index(drop=True)

def _params(**kw):
    base = dict(input_path="mem", output_dir="out", block_x=5, block_y=5,
                block_z=5, cutoff=1.0, smu_x=10.0, step_x=5.0, oxide_col=None,
                ore_rule="value", audit_fields=True)
    base.update(kw)
    return DilutionParams(**base)

def test_single_high_block_is_diluted_to_smu():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    ob = run_dilution(_params(), df=df).ore_blocks
    assert len(ob) == 1
    assert ob.XINC.iloc[0] == 10.0
    assert ob.AU.iloc[0] == pytest.approx(2.5)
    assert ob.TONNES.iloc[0] == pytest.approx(10 * 5 * 5 * 2.0)

def test_narrow_waste_between_ore_is_taken_as_ore():
    df = _strip_model([0, 0, 3, 3, 0, 3, 3, 0, 0])
    ob = run_dilution(_params(), df=df).ore_blocks
    assert ob.ZONE_ID.nunique() == 1
    assert list(ob.XINC) == [10.0, 15.0]
    assert ob.AU.iloc[1] == pytest.approx(2.0)

def test_mii_vs_mi_definition_and_reporting():
    df = _strip_model([0, 2, 2, 2, 2, 0], [2, 2, 3, 3, 2, 2])
    mii = run_dilution(_params(mode="MII"), df=df).ore_blocks
    assert mii.TONNES.sum() == pytest.approx(4 * 250)
    assert (mii.AU == 2.0).all()
    mi = run_dilution(_params(mode="MI"), df=df).ore_blocks
    assert len(mi) == 0

def test_rescat4_reported_at_true_grade():
    df = _strip_model([0, 2, 2, 4, 2, 2, 0], [2, 2, 2, 4, 2, 2, 2])
    ob = run_dilution(_params(mode="MII"), df=df).ore_blocks
    assert ob.XINC.sum() == 25.0
    total_au = (ob.AU * ob.TONNES).sum() / ob.TONNES.sum()
    assert total_au == pytest.approx((2 * 4 + 4) / 5)
    assert ob["TONNES_RESCAT_4"].sum() == pytest.approx(250)

def test_short_waste_allowed_at_model_edge_not_against_void():

    edge = run_dilution(_params(), df=_strip_model([0, 3, 3, 3])).ore_blocks
    assert edge.X_FROM.min() == 5.0

    df = _strip_model([0, 0, 0, 0, 0, 3, 3, 3, 0, 0], skip=[3])
    ob = run_dilution(_params(), df=df).ore_blocks
    assert ob.X_FROM.min() == 20.0

def test_zero_density_is_void():
    df = _strip_model([0, 0, 0, 0, 0, 3, 3, 3, 0, 0])
    df.loc[3, "DENSITY"] = 0.0
    res = run_dilution(_params(), df=df)
    assert res.ore_blocks.X_FROM.min() == 20.0
    assert res.block_model.ORE_FRAC.iloc[3] == 0

def test_oxide_dominant_by_mass():
    df = _strip_model([0, 3, 3, 3, 0], oxide=["OX", "OX", "FR", "FR", "FR"])
    df.loc[1, "DENSITY"] = 1.0
    ob = run_dilution(_params(oxide_col="OXIDE", smu_x=15.0), df=df).ore_blocks
    assert len(ob) == 1
    assert ob.OXIDE.iloc[0] == "FR"
    assert ob.OXIDE_PCT.iloc[0] == pytest.approx(100 * 500 / 625)

def test_smu_ratio_check():
    assert check_smu_ratio("X", 10, 5) == 5
    assert check_smu_ratio("X", 2.5, 10) == 2.5
    assert check_smu_ratio("X", 7.5, 10) == 2.5
    assert check_smu_ratio("Y", 7.5, 5) == 2.5
    with pytest.raises(ValueError):
        check_smu_ratio("X", 7.123, 5)
    with pytest.raises(ValueError):
        run_dilution(_params(smu_y=3.001), df=_strip_model([0, 3, 3]))
    with pytest.raises(ValueError):
        _params(smu_x=10, step_x=3).validate()

def test_smu_smaller_than_parent_subcells_strips():

    df = pd.DataFrame({"XC": [5, 15, 25, 35], "YC": 5, "ZC": 5,
                       "AU": [0, 3, 3, 0], "RESCAT": 1, "DENSITY": 2.0})
    res = run_dilution(_params(block_x=10, block_y=10, block_z=10, smu_x=10,
                               smu_y=5, smu_z=5, step_x=5), df=df)
    ob = res.ore_blocks
    assert len(ob) == 8
    assert set(ob.YC) == {2.5, 7.5} and set(ob.ZC) == {2.5, 7.5}
    assert ob.TONNES.sum() == pytest.approx(2 * 1000 * 2.0)
    assert res.diluted_model.TONNES.sum() == pytest.approx(4 * 1000 * 2.0)

def test_block_model_flag():
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    res = run_dilution(_params(step_x=2.5), df=df)
    assert res.block_model.ORE_FRAC.sum() == pytest.approx(2.0)

def test_sub_blocked_model_matches_regular():
    grades = [0, 0, 3, 1, 3, 0, 0, 0]
    reg = run_dilution(_params(), df=_strip_model(grades)).ore_blocks
    rows = []
    for i, g in enumerate(grades):
        for sx in (1.25, 3.75):
            for sy in (1.25, 3.75):
                rows.append(dict(XC=i * 5 + sx, YC=sy, ZC=2.5, XINC=2.5,
                                 YINC=2.5, ZINC=5.0, AU=g, RESCAT=2, DENSITY=2.0))
    sub = run_dilution(_params(xinc_col="XINC", yinc_col="YINC", zinc_col="ZINC"),
                       df=pd.DataFrame(rows)).ore_blocks
    assert list(sub.XINC) == list(reg.XINC)
    assert np.allclose(sub.AU, reg.AU) and np.allclose(sub.TONNES, reg.TONNES)

def test_outputs_written(tmp_path):
    df = _strip_model([0, 3, 3, 0, 0, 3, 3, 3, 0], oxide=["OX"] * 9)
    res = run_dilution(_params(oxide_col="OXIDE"), df=df)
    paths = write_outputs(res, str(tmp_path))
    for pth in paths.values():
        assert os.path.getsize(pth) > 0
    lines = open(paths["strings"]).read().splitlines()
    assert lines[-1].endswith("END")
    pts = [ln for ln in lines[2:] if ln.startswith("1,")]
    assert len(pts) == 5 * len(res.ore_blocks)
    y, x = map(float, pts[0].split(",")[1:3])
    assert (x, y) == (res.ore_blocks.X_FROM.iloc[0], 0.0)
    assert open(paths["dxf"]).read().count("POLYLINE") == len(res.ore_blocks)
    dm = res.diluted_model
    assert dm.TONNES.sum() == pytest.approx(df.shape[0] * 250)
    assert set(dm.MATERIAL) == {"ORE", "WASTE"}

def test_diluted_model_subcells_parent_grid():

    rng = np.random.default_rng(4)
    xs, ys = np.meshgrid(np.arange(12) * 10 + 5, np.arange(4) * 10 + 5)
    df = pd.DataFrame({"XC": xs.ravel(), "YC": ys.ravel(), "ZC": 5.0,
                       "AU": rng.lognormal(0, 1, xs.size), "RESCAT": 2,
                       "DENSITY": 2.5})
    res = run_dilution(_params(block_x=10, block_y=10, block_z=10, smu_x=5,
                               smu_y=20, smu_z=10, step_x=2.5), df=df)
    dm, ob = res.diluted_model, res.ore_blocks
    assert len(ob) and (ob.AU_MII >= 1.0 - 1e-9).all()
    assert np.array_equal(np.floor(dm.X_FROM / 10), np.floor((dm.X_TO - 1e-6) / 10))
    assert set(dm.YINC) == {10.0} and set(np.mod(dm.YC, 10)) == {5.0}
    assert dm.TONNES.sum() == pytest.approx(len(df) * 1000 * 2.5)
    ore_dm = dm[dm.MATERIAL == "ORE"]
    assert ore_dm.METAL_G.sum() == pytest.approx(ob.METAL_G.sum())
    assert ore_dm.groupby("BLOCK_ID").D_AU.nunique().max() == 1
    assert {"D_AU", "AU_MII", "D_OXIDE", "D_OXIDE_PCT"} <= set(dm.columns)
    assert "AU" not in dm.columns

def test_step_must_fit_parent():
    with pytest.raises(ValueError):
        run_dilution(_params(smu_x=10, step_x=2.0, block_x=5), df=_strip_model([0, 3]))

def _subcelled_model():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(12):
        for j in range(2):
            px, py, pz = 1000 + i * 10, 2000 + j * 10, 100
            au = rng.lognormal(-0.3, 1.2)
            if i % 2 == 0:
                for a in range(4):
                    for b in range(2):
                        for c in range(2):
                            if i == 0 and a < 2:
                                continue
                            rows.append(dict(XC=px + a * 2.5 + 1.25, YC=py + b * 5 + 2.5,
                                             ZC=pz + c * 5 + 2.5, XINC=2.5, YINC=5, ZINC=5,
                                             AU=au * rng.lognormal(0, 0.5), RESCAT=2,
                                             DENSITY=2.6))
            else:
                rows.append(dict(XC=px + 5, YC=py + 5, ZC=pz + 5, XINC=10, YINC=10,
                                 ZINC=10, AU=au, RESCAT=2, DENSITY=2.6))
    return pd.DataFrame(rows)

def test_diluted_model_keeps_original_parent_grid():
    df = _subcelled_model()
    res = run_dilution(_params(block_x=None, block_y=None, block_z=None, smu_x=5,
                               smu_y=5, smu_z=5, step_x=2.5, xinc_col="XINC",
                               yinc_col="YINC", zinc_col="ZINC"), df=df)
    p, dm = res.params, res.diluted_model
    assert (p["block_x"], p["block_y"], p["block_z"]) == (10, 10, 10)
    assert (p["origin_x"], p["origin_y"], p["origin_z"]) == (1000, 2000, 100)
    assert (dm.PARENT_XINC == 10).all() and (dm.NX == 12).all()

    for c, inc, o in (("XC", "XINC", 1000), ("YC", "YINC", 2000), ("ZC", "ZINC", 100)):
        lo = np.floor((dm[c] - dm[inc] / 2 - o) / 10 + 1e-9)
        hi = np.floor((dm[c] + dm[inc] / 2 - o) / 10 - 1e-9)
        assert (lo == hi).all()
    i = np.floor((dm.XC - 1000) / 10).astype(int)
    j = np.floor((dm.YC - 2000) / 10).astype(int)
    assert (dm.IJK == j * 12 + i).all()
    assert dm.TONNES.sum() == pytest.approx((df.XINC * df.YINC * df.ZINC * 2.6).sum())

def test_given_origin_below_model_is_used():
    df = _strip_model([0, 3, 3, 0])
    res = run_dilution(_params(origin_x=-10.0), df=df)
    assert res.params["origin_x"] == -10.0
    assert res.params["parent_nx"] == 6
    with pytest.raises(ValueError):
        run_dilution(_params(origin_x=1.0), df=df)

def test_threads_match_single_thread():
    df = _subcelled_model()
    kw = dict(block_x=None, block_y=None, block_z=None, smu_x=5, smu_y=5,
              smu_z=5, step_x=2.5, xinc_col="XINC", yinc_col="YINC",
              zinc_col="ZINC")
    one = run_dilution(_params(threads=1, **kw), df=df)
    two = run_dilution(_params(threads=2, **kw), df=df)
    pd.testing.assert_frame_equal(one.ore_blocks, two.ore_blocks)
    pd.testing.assert_frame_equal(one.diluted_model, two.diluted_model)
    assert np.allclose(one.block_model.ORE_FRAC, two.block_model.ORE_FRAC)

def _block_model_3d(seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(8):
        for j in range(6):
            for k in range(3):
                rows.append(dict(XC=5 + 10 * i, YC=5 + 10 * j, ZC=5 + 10 * k,
                                 AU=float(rng.lognormal(-0.2, 1.0)), RESCAT=2,
                                 DENSITY=float(rng.uniform(2.2, 3.0))))
    return pd.DataFrame(rows)

@pytest.mark.parametrize("smu", [(7.5, 10, 10), (10, 7.5, 10), (10, 10, 7.5),
                                 (7.5, 7.5, 7.5), (15, 15, 15)])
def test_smu_crossing_parent_boundaries_reconciles(smu):
    df = _block_model_3d()
    sx, sy, sz = smu
    p = _params(block_x=10, block_y=10, block_z=10, smu_x=sx, smu_y=sy, smu_z=sz,
                step_x=2.5, cutoff=1.0)
    res = run_dilution(p, df=df)
    dm, ob = res.diluted_model, res.ore_blocks
    assert len(ob)

    for c, inc, o in (("XC", "XINC", 0), ("YC", "YINC", 0), ("ZC", "ZINC", 0)):
        lo = np.floor((dm[c] - dm[inc] / 2 - o) / 10 + 1e-9)
        hi = np.floor((dm[c] + dm[inc] / 2 - o) / 10 - 1e-9)
        assert (lo == hi).all(), c

    df["IJK"] = ((np.floor(df.ZC / 10) * 6 + np.floor(df.YC / 10)) * 8
                 + np.floor(df.XC / 10)).astype(int)
    t_in = (df.DENSITY * 1000).groupby(df.IJK).sum()
    t_out = dm.groupby("IJK").TONNES.sum()
    pd.testing.assert_series_equal(t_out.sort_index(), t_in.sort_index(),
                                   check_names=False, check_index_type=False)

    ore = dm[dm.MATERIAL == "ORE"]
    g = ore.groupby("BLOCK_ID").agg(T=("TONNES", "sum"), M=("METAL_G", "sum"))
    blocks = ob.set_index("BLOCK_ID")
    assert np.allclose(g["T"], blocks.TONNES[g.index])
    assert np.allclose(g["M"], blocks.METAL_G[g.index])
    assert np.allclose(ore.D_AU, blocks.AU[ore.BLOCK_ID].to_numpy())

def _attr_model(grades, attr, bx=5.0):
    n = len(grades)
    return pd.DataFrame({
        "XC": np.arange(n) * bx + bx / 2, "YC": 2.5, "ZC": 2.5,
        "AU": grades, "RESCAT": [2] * n, "DENSITY": 2.0, "WEATH": attr,
    })

def test_partial_dilution_off_unchanged():
    grades = [0, 0, 3, 3, 0, 3, 3, 0, 0]
    attr = ["OX"] * 4 + ["FR"] * 5
    df = _attr_model(grades, attr)
    r1 = run_dilution(_params(), df=df)
    r2 = run_dilution(_params(dilute_attr_col=None, dilute_attr_values=[]), df=df)
    pd.testing.assert_frame_equal(r1.ore_blocks, r2.ore_blocks)
    assert r1.summary.attrs["dilution_pct"] == r2.summary.attrs["dilution_pct"]

def test_partial_dilution_selected_blocks_diluted():

    grades = [0, 0, 3, 3, 0, 0, 0, 0]
    attr = ["OX", "OX", "OX", "OX", "FR", "FR", "FR", "FR"]
    df = _attr_model(grades, attr)

    r_no = run_dilution(_params(), df=df)

    r = run_dilution(_params(dilute_attr_col="WEATH",
                             dilute_attr_values=["OX"]), df=df)
    ob = r.ore_blocks
    assert len(ob) >= 1
    assert ob.AU.iloc[0] == r_no.ore_blocks.AU.iloc[0]
    assert r.summary.attrs.get("partial_dilution") is True

def test_partial_dilution_unselected_becomes_waste():

    grades = [0, 3, 3, 0]
    attr = ["FR", "FR", "FR", "FR"]
    df = _attr_model(grades, attr)
    r_all = run_dilution(_params(), df=df)
    r_ox = run_dilution(_params(dilute_attr_col="WEATH",
                                dilute_attr_values=["OX"]), df=df)
    assert len(r_all.ore_blocks) >= 1

    assert len(r_ox.ore_blocks) == 0

def test_partial_dilution_summary_uses_selected_totals():
    grades = [0, 0, 3, 3, 0, 0, 0, 0]
    attr = ["OX", "OX", "OX", "OX", "FR", "FR", "FR", "FR"]
    df = _attr_model(grades, attr)
    r = run_dilution(_params(dilute_attr_col="WEATH",
                             dilute_attr_values=["OX"]), df=df)
    assert r.summary.attrs.get("partial_dilution") is True

def test_dm_export_parent_increments(tmp_path):
    from dilution.io_dm import read_dm, _fields, WORDS_PER_PAGE
    rng = np.random.default_rng(42)
    rows = []
    for i in range(4):
        for j in range(3):
            px, py, pz = i * 15, j * 15, 0
            for sz in range(4):
                rows.append(dict(
                    XC=px + 11.25 / 2, YC=py + 11.25 / 2,
                    ZC=pz + sz * 2.25 + 2.25 / 2,
                    XINC=11.25, YINC=11.25, ZINC=2.25,
                    AU=rng.lognormal(0, 1), RESCAT=2, DENSITY=2.7))
    df = pd.DataFrame(rows)
    res = run_dilution(_params(
        block_x=15, block_y=15, block_z=9, smu_x=15,
        smu_y=11.25, smu_z=2.25, step_x=15,
        xinc_col="XINC", yinc_col="YINC", zinc_col="ZINC"), df=df)
    paths = write_outputs(res, str(tmp_path))
    assert "diluted_model_dm" in paths
    dm_path = paths["diluted_model_dm"]
    dm = read_dm(dm_path)
    assert len(dm) > 0
    assert set(dm.XINC.unique()) <= {11.25, 15.0}
    assert dm.XMORIG.iloc[0] == res.params["origin_x"]
    assert dm.NX.iloc[0] == res.params["parent_nx"]
    with open(dm_path, "rb") as fh:
        fields, ws, *_ = _fields(fh.read(WORDS_PER_PAGE * 8))
    xinc_f = fields["XINC"]
    stored = [p for _, p in xinc_f["words"] if p > 0]
    implicit = [p for _, p in xinc_f["words"] if p == 0]
    assert stored, "XINC must have a stored (per-record) position"
    assert implicit, "XINC must have an implicit (header) position for the parent increment"
    assert xinc_f["default"] == pytest.approx(15.0)
    assert fields["YINC"]["default"] == pytest.approx(15.0)
    assert fields["ZINC"]["default"] == pytest.approx(9.0)
    csv = pd.read_csv(paths["diluted_model"])
    m = dm.merge(csv, on=["XC", "YC", "ZC"], suffixes=("", "_csv"))
    assert len(m) == len(csv)
    grade = "D_AU" if "D_AU" in csv else "AU"
    assert np.allclose(m[grade], m[grade + "_csv"])
