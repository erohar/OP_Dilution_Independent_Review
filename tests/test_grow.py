import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dilution.checks import check_outputs
from dilution.core import DilutionParams, format_report, run_dilution
from dilution.grow import (FREE, VOID, WASTE, allowance, narrow_waste,
                           row_ok)

def _model(grade, bx=5.0, by=1.0, bz=5.0):
    g = np.asarray(grade, dtype=float)
    r, k = np.nonzero(np.ones_like(g, dtype=bool))
    return pd.DataFrame({"XC": k * bx + bx / 2, "YC": r * by + by / 2, "ZC": bz / 2,
                         "AU": g[r, k], "RESCAT": 2, "DENSITY": 2.0})

def _params(**kw):
    base = dict(input_path="mem", output_dir="out", block_x=5, block_y=1, block_z=5,
                cutoff=1.0, smu_x=10.0, smu_y=4.0, step_x=5.0, step_y=1.0,
                oxide_col=None, audit_fields=True)
    base.update(kw)
    return DilutionParams(**base)

def _grid(rows, cols, cells):
    g = np.zeros((rows, cols))
    for r, k, v in cells:
        g[r, k] = v
    return g

def test_block_grows_up_into_partial_ore_of_next_strip():

    g = _grid(12, 8, [(r, k, 3.0) for r in range(5) for k in (2, 3)])
    res = run_dilution(_params(), df=_model(g))
    ob = res.ore_blocks
    assert len(ob) == 1
    assert (ob.YC.iloc[0], ob.YINC.iloc[0]) == (2.5, 5.0)
    assert ob.TONNES.iloc[0] == pytest.approx(10 * 5 * 5 * 2.0)
    assert res.summary.attrs["ore_loss_pct"] == pytest.approx(0.0)
    assert res.stats["grown"] == 1

    flat = run_dilution(_params(step_y=None), df=_model(g))
    assert flat.ore_blocks.YINC.iloc[0] == 4.0
    assert flat.ore_blocks.METAL_G.sum() < ob.METAL_G.sum()

def test_block_grows_down_into_partial_ore_of_strip_below():
    g = _grid(12, 8, [(r, k, 3.0) for r in range(7, 12) for k in (2, 3)])
    ob = run_dilution(_params(), df=_model(g)).ore_blocks
    assert len(ob) == 1
    assert (ob.YC.iloc[0], ob.YINC.iloc[0]) == (9.5, 5.0)

def test_growth_keeps_block_above_cutoff_and_takes_most_ore():

    g = _grid(12, 8, [(r, k, 3.0) for r in range(5) for k in (2, 3)])
    g[5, 2:4] = 0.9
    ob = run_dilution(_params(), df=_model(g)).ore_blocks
    assert ob.YINC.iloc[0] == 5.0

    g = _grid(12, 8, [(r, k, 1.0) for r in range(4) for k in (2, 3)])
    g[4, 2:4] = 3.0
    ob = run_dilution(_params(), df=_model(g)).ore_blocks
    assert ob.YINC.iloc[0] == 5.0
    g[4, 2:4] = 0.5
    ob = run_dilution(_params(), df=_model(g)).ore_blocks
    assert ob.YINC.iloc[0] == 4.0

def test_growth_is_under_one_smu():

    g = _grid(12, 8, [(r, k, 3.0) for r in range(4) for k in (2, 3)])
    g[4:7, 2:4] = 1.2
    ob = run_dilution(_params(), df=_model(g)).ore_blocks
    assert ob.YINC.iloc[0] == 7.0

def test_no_growth_leaving_narrow_waste_in_y():

    g = _grid(12, 8, [(r, k, 3.0) for r in list(range(5)) + list(range(8, 12))
                      for k in (2, 3)])
    ob = run_dilution(_params(), df=_model(g)).ore_blocks
    assert sorted(ob.YINC) == [4.0, 4.0]

def test_no_growth_leaving_narrow_waste_in_x():

    g = _grid(12, 10, [(r, k, 3.0) for r in range(4) for k in (2, 3)])
    g[4, 2:4] = 2.0
    g[4:8, 5:7] = 3.0
    ob = run_dilution(_params(ore_rule="value"), df=_model(g)).ore_blocks
    assert sorted(ob.YINC) == [4.0, 4.0]

    g = _grid(12, 10, [(r, k, 3.0) for r in range(4) for k in (2, 3)])
    g[4, 2:4] = 2.0
    g[4:8, 6:8] = 3.0
    ob = run_dilution(_params(ore_rule="value"), df=_model(g)).ore_blocks.sort_values("X_FROM")
    assert list(ob.YINC) == [5.0, 4.0]

def test_growth_shares_waste_between_ore_on_both_sides():

    g = _grid(16, 8, [(r, k, 3.0) for r in list(range(4)) + list(range(12, 16))
                      for k in (2, 3)])
    g[4:7, 2:4] = 1.3
    g[9:12, 2:4] = 1.3
    ob = run_dilution(_params(), df=_model(g)).ore_blocks.sort_values("YC")
    assert list(ob.YINC) == [6.0, 6.0]

def test_value_rule_grows_only_for_metal():
    g = _grid(12, 8, [(r, k, 3.0) for r in range(5) for k in (2, 3)])
    ob = run_dilution(_params(ore_rule="value"), df=_model(g)).ore_blocks
    assert ob.YINC.iloc[0] == 5.0

def test_edge_dilution_in_x_and_y():
    g = _grid(16, 8, [(r, k, 3.0) for r in range(4, 8) for k in (2, 3)])
    with pytest.raises(ValueError, match="whole number of X steps"):
        res = run_dilution(_params(step_x=5.0, elos=1.0), df=_model(g))
    res = run_dilution(_params(block_x=1.0, step_x=1.0, elos=1.0),
                       df=_model(np.repeat(g, 5, axis=1), bx=1.0))
    ob = res.ore_blocks
    assert len(ob) == 1

    assert (ob.XINC.iloc[0], ob.YINC.iloc[0]) == (12.0, 6.0)
    assert (ob.X_FROM.iloc[0], ob.YC.iloc[0]) == (9.0, 6.0)
    assert ob.ELOS_TONNES.iloc[0] == pytest.approx((12 * 6 - 10 * 4) * 5 * 2.0)
    assert ob.AU_MII_DESIGN.iloc[0] == pytest.approx(3.0)

def test_legacy_edge_dilution_in_x_only():
    g = _grid(16, 8, [(r, k, 3.0) for r in range(4, 8) for k in (2, 3)])
    df = _model(np.repeat(g, 5, axis=1), bx=1.0)

    with pytest.raises(ValueError, match="whole number of Y steps"):
        _params(block_x=1.0, step_x=0.5, elos=0.5).validate()
    _params(block_x=1.0, step_x=0.5, elos=0.5, elos_axes="x").validate()
    res = run_dilution(_params(block_x=1.0, step_x=1.0, elos=1.0, elos_axes="x"), df=df)
    ob = res.ore_blocks
    assert len(ob) == 1
    assert (ob.XINC.iloc[0], ob.YINC.iloc[0]) == (12.0, 4.0)
    assert ob.ELOS_TONNES.iloc[0] == pytest.approx(2 * 4 * 5 * 2.0)
    assert "X only" in format_report(res)
    with pytest.raises(ValueError, match="axes"):
        _params(elos_axes="y").validate()

def test_x_only_edge_dilution_locks_blocks_to_smu_y():

    g = _grid(12, 8, [(r, k, 3.0) for r in range(5) for k in (2, 3)])
    df = _model(np.repeat(g, 5, axis=1), bx=1.0)
    kw = dict(block_x=1.0, step_x=1.0, elos=1.0)
    xy = run_dilution(_params(**kw), df=df)
    xo = run_dilution(_params(elos_axes="x", **kw), df=df)
    xy2 = run_dilution(_params(**kw), df=df)
    ob = xo.ore_blocks
    assert xo.stats["grown"] == 0
    assert xo.summary.attrs["t_grown"] == 0.0
    assert (ob.YINC == 4.0).all() and ob.YC.iloc[0] == 2.0
    assert ob.XINC.iloc[0] == 12.0
    assert ob.ELOS_TONNES.iloc[0] == pytest.approx(2 * 4 * 5 * 2.0)
    assert (xo.stats["t_elos_x"], xo.stats["t_elos_y"]) == (80.0, 0.0)
    assert (xy.stats["t_elos_x"], xy.stats["t_elos_y"]) == (100.0, 120.0)

    legacy = run_dilution(_params(step_y=None, elos_axes="x", **kw), df=df).ore_blocks
    cols = ["XC", "YC", "XINC", "YINC", "TONNES", "METAL_G", "ELOS_TONNES"]
    pd.testing.assert_frame_equal(ob[cols].reset_index(drop=True),
                                  legacy[cols].reset_index(drop=True))

    for res in (xy, xy2):
        o = res.ore_blocks
        assert res.stats["grown"] == 1
        assert res.summary.attrs["t_grown"] == pytest.approx(100.0)
        assert (o.YC.iloc[0], o.YINC.iloc[0], o.XINC.iloc[0]) == (3.0, 6.0, 12.0)
        assert o.TONNES.iloc[0] == pytest.approx(720.0)
        assert o.ELOS_TONNES.iloc[0] == pytest.approx(220.0)

def test_no_edge_dilution_between_blocks_touching_in_y():
    g = np.zeros((16, 40))
    g[4:12, 10:20] = 3.0
    res = run_dilution(_params(block_x=1.0, step_x=1.0, elos=1.0), df=_model(g, bx=1.0))
    ob = res.ore_blocks.sort_values("YC")
    assert len(ob) == 2

    assert list(ob.YINC) == [5.0, 5.0]
    assert list(ob.YC) == [5.5, 10.5]

def _random_model(seed, nr=24, nk=30):
    rng = np.random.default_rng(seed)
    g = np.zeros((nr, nk))
    for _ in range(4):
        r0, k0 = rng.integers(0, nr - 3), rng.integers(0, nk - 4)
        g[r0:r0 + rng.integers(2, 9), k0:k0 + rng.integers(2, 10)] += rng.uniform(0.5, 4)
    g += rng.uniform(0, 0.6, g.shape)
    df = _model(g)
    return df.drop(index=rng.choice(len(df), 6, replace=False)).reset_index(drop=True)

@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("rule", ["recover", "value"])
def test_random_models_keep_every_rule(seed, rule):
    df = _random_model(seed)
    elos = 5.0 if seed % 3 == 0 else 0.0
    p = _params(ore_rule=rule, elos=elos, smu_y=5.0 if elos else 4.0)
    res = run_dilution(p, df=df)
    bad = [c for c in check_outputs(df, p, res) if c.status == "FAIL"]
    assert not bad, bad
    assert res.stats["narrow_waste"] == 0
    dm = res.diluted_model
    assert not dm.duplicated(["XC", "YC", "ZC"]).any()

    ob = res.ore_blocks
    for i in range(len(ob)):
        a = ob.iloc[i]
        for j in range(i + 1, len(ob)):
            b = ob.iloc[j]
            assert (min(a.X_TO, b.X_TO) <= max(a.X_FROM, b.X_FROM) + 1e-9
                    or abs(a.YC - b.YC) >= (a.YINC + b.YINC) / 2 - 1e-9)

    assert np.allclose(ob.YINC, np.round(ob.YINC))
    assert (ob.YINC >= p.smu_y - 1e-9).all()
    assert (ob.YINC <= 2 * p.smu_y - 1.0 + 2 * elos + 1e-9).all()

    small = run_dilution(p, df=df, job_cells=150)
    assert small.stats["jobs"] > res.stats["jobs"]
    key = ["ZC", "YC", "X_FROM"]
    a = ob.drop(columns=["BLOCK_ID", "ZONE_ID"]).sort_values(key).reset_index(drop=True)
    b = (small.ore_blocks.drop(columns=["BLOCK_ID", "ZONE_ID"])
         .sort_values(key).reset_index(drop=True))
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)
    assert small.summary.TONNES.to_numpy() == pytest.approx(res.summary.TONNES.to_numpy())

def test_y_step_must_divide_smu_y():
    with pytest.raises(ValueError, match="whole number of Y steps"):
        run_dilution(_params(step_y=3.0), df=_model(np.ones((4, 4))))
    with pytest.raises(ValueError):
        _params(step_y=0.0).validate()

def test_allowance_shares_and_edges():
    W, ORE = WASTE, 0
    col = np.array([[FREE], [ORE], [W], [W], [W], [W], [W], [ORE], [W], [W], [FREE]])
    lo, hi = allowance(col, 2, void_free=False)

    assert (lo[2, 0], hi[6, 0]) == (2, 1)

    assert lo[8, 0] == 2
    col[10, 0] = VOID
    lo, _ = allowance(col, 2, void_free=False)
    assert lo[8, 0] == 0
    lo, _ = allowance(col, 2, void_free=True)
    assert lo[8, 0] == 2

def test_row_ok():
    W = WASTE
    lab = np.array([[FREE, W, W, W, W, W, 3, W, W, VOID]])
    assert row_ok(lab, 0, 1, 3, 2, False)
    assert row_ok(lab, 0, 1, 4, 2, False)
    assert not row_ok(lab, 0, 2, 5, 2, False)
    assert row_ok(lab, 0, 3, 6, 2, False)
    assert not row_ok(lab, 0, 7, 8, 2, False)
    assert row_ok(lab, 0, 7, 8, 2, True)
    assert not row_ok(lab, 0, 5, 7, 2, False)

def test_narrow_waste_counts():
    W = WASTE
    lab = np.array([[0, W, 1, W, W, 2]])
    assert narrow_waste(lab, 2, 2, np.array([0])) == 1

@pytest.mark.parametrize("seed", range(3))
def test_job_split_with_growth_on_many_strips(seed):

    rng = np.random.default_rng(1000 + seed)
    g = np.zeros((100, 40))
    for _ in range(25):
        r0, k0 = rng.integers(0, 97), rng.integers(0, 36)
        g[r0:r0 + rng.integers(2, 9), k0:k0 + rng.integers(2, 12)] += rng.uniform(0.5, 4)
    g += rng.uniform(0, 0.7, g.shape)
    df = _model(g)
    elos = 5.0 if seed % 2 else 0.0
    p = _params(elos=elos, smu_y=5.0 if elos else 4.0)
    a = run_dilution(p, df=df)
    b = run_dilution(p, df=df, job_cells=100)
    assert a.stats["grown"] > 0 and b.stats["jobs"] >= 10
    key = ["ZC", "YC", "X_FROM"]
    oa = a.ore_blocks.drop(columns=["BLOCK_ID", "ZONE_ID"]).sort_values(key)
    ob = b.ore_blocks.drop(columns=["BLOCK_ID", "ZONE_ID"]).sort_values(key)
    pd.testing.assert_frame_equal(oa.reset_index(drop=True), ob.reset_index(drop=True),
                                  check_exact=False, rtol=1e-9)
    assert a.ore_frac == pytest.approx(b.ore_frac)
    da = a.diluted_model.drop(columns=["BLOCK_ID", "ZONE_ID"]).sort_values(["YC", "XC"])
    db = b.diluted_model.drop(columns=["BLOCK_ID", "ZONE_ID"]).sort_values(["YC", "XC"])
    pd.testing.assert_frame_equal(da.reset_index(drop=True), db.reset_index(drop=True),
                                  check_exact=False, rtol=1e-9)
