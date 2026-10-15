import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dilution.core import (DilutionParams, apply_elos,
                           block_outline, run_dilution, write_outputs)
from dilution.regions import (as_regions, load_regions,
                              make_region, order_regions, owner)

def _rect(name, cx, cy, length, width, azimuth, priority=1):
    a = math.radians(azimuth)
    u = np.array([math.sin(a), math.cos(a)])
    v = np.array([u[1], -u[0]])
    o = np.array([cx, cy]) - u * length / 2 - v * width / 2
    pts = [o, o + u * length, o + u * length + v * width, o + v * width]
    return dict(name=name, corners=[tuple(map(float, p)) for p in pts],
                priority=priority)

def _grid_model(nx=24, ny=12, nz=2, b=5.0, seed=3):
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(np.arange(nx) * b + b / 2, np.arange(ny) * b + b / 2,
                          np.arange(nz) * b + b / 2, indexing="ij")
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    d1 = np.abs((x - 30) - 0.5 * y)
    d2 = np.abs((x - 85) - 0.3 * y)
    au = np.where((d1 < 6) | (d2 < 4), rng.uniform(1.0, 4.0, x.size),
                  rng.uniform(0.0, 0.6, x.size))
    return pd.DataFrame({"XC": x, "YC": y, "ZC": z, "AU": au.round(3),
                         "RESCAT": rng.integers(1, 5, x.size),
                         "DENSITY": 2.6})

def _strip(grades, bx=5.0):
    n = len(grades)
    return pd.DataFrame({"XC": np.arange(n) * bx + bx / 2, "YC": 2.5, "ZC": 2.5,
                         "AU": grades, "RESCAT": 2, "DENSITY": 2.0})

def _params(**kw):
    base = dict(input_path="mem", output_dir="out", block_x=5, block_y=5,
                block_z=5, cutoff=1.0, smu_x=10.0, step_x=5.0, oxide_col=None)
    base.update(kw)
    return DilutionParams(**base)

def test_region_geometry_and_azimuth():
    r = make_region("A", [(0, 0), (10, 0), (10, -4), (0, -4)])
    assert r.azimuth == pytest.approx(90.0)
    assert (r.length, r.width) == (pytest.approx(10), pytest.approx(4))
    assert r.contains(np.array([5.0, 11.0]), np.array([-2.0, -2.0])).tolist() == [True, False]
    rot = as_regions([_rect("B", 0, 0, 20, 10, 30)])[0]
    assert rot.azimuth == pytest.approx(30.0)
    u, v = rot.to_local(*rot.to_world(3.0, 4.0))
    assert (float(u), float(v)) == (pytest.approx(3.0), pytest.approx(4.0))

def test_region_rejects_bad_shapes_and_drops_closing_point():
    closed = make_region("A", [(0, 0), (10, 0), (10, 5), (0, 5), (0, 0)])
    assert len(closed.corners) == 4
    with pytest.raises(ValueError, match="not a rectangle"):
        make_region("B", [(0, 0), (10, 0), (12, 5), (0, 5)])
    with pytest.raises(ValueError, match="4 corner points"):
        make_region("C", [(0, 0), (10, 0), (10, 5)])

def test_overlapping_regions_need_distinct_priorities():
    a = as_regions([_rect("A", 0, 0, 20, 20, 90, 1)])[0]
    b = as_regions([_rect("B", 5, 5, 20, 20, 45, 1)])[0]
    far = as_regions([_rect("C", 100, 100, 20, 20, 45, 1)])[0]
    with pytest.raises(ValueError, match="overlap"):
        order_regions([a, b])
    assert [r.name for r in order_regions([a, far])] == ["A", "C"]
    with pytest.raises(ValueError, match="reserved"):
        order_regions(as_regions([_rect("MODEL", 0, 0, 5, 5, 90)]))

def test_owner_takes_highest_priority():
    regions = order_regions(as_regions([_rect("LOW", 0, 0, 20, 20, 90, 2),
                                        _rect("HIGH", 5, 0, 20, 20, 60, 1)]))
    who = owner(np.array([[5.0, -8.0], [50.0, 0.0]]),
                np.array([[0.0, 0.0], [0.0, 0.0]]), regions)
    assert who.shape == (2, 2)
    assert [regions[i].name if i >= 0 else None for i in who.ravel()] == \
        ["HIGH", "LOW", None, "HIGH"]

def test_load_regions_formats(tmp_path):
    wide = tmp_path / "wide.csv"
    wide.write_text("REGION,PRIORITY,X1,Y1,X2,Y2,X3,Y3,X4,Y4\n"
                    "A,2,0,0,10,0,10,-5,0,-5\nB,1,20,0,30,0,30,-5,20,-5\n")
    long = tmp_path / "long.csv"
    long.write_text("REGION,X,Y\n" + "".join(
        f"{n},{x},{y}\n" for n, off in (("A", 0), ("B", 20))
        for x, y in ((off, 0), (off + 10, 0), (off + 10, -5), (off, -5))))
    surpac = tmp_path / "r.str"
    surpac.write_text("r,01-Jan-26,,\n0, 0.000, 0.000, 0.000,\n"
                      + "".join(f"1,{y},{x},0,2\n" for x, y in
                                ((0, 0), (10, 0), (10, -5), (0, -5), (0, 0)))
                      + "0, 0.000, 0.000, 0.000,\n"
                      + "".join(f"3,{y},{x},0,1\n" for x, y in
                                ((20, 0), (30, 0), (30, -5), (20, -5)))
                      + "0, 0.000, 0.000, 0.000, END\n")
    for path in (wide, long, surpac):
        regions = order_regions(load_regions(str(path)))
        assert len(regions) == 2
        assert all(r.length == pytest.approx(10) for r in regions)
    assert [r.name for r in order_regions(load_regions(str(wide)))] == ["B", "A"]
    with pytest.raises(FileNotFoundError):
        load_regions(str(tmp_path / "none.csv"))

@pytest.mark.parametrize("blocks,n,m,e,want", [

    ([(10, 20)], 40, 10, 2, [(8, 22)]),

    ([(1, 11)], 20, 10, 3, [(0, 14)]),

    ([(10, 20), (20, 30)], 50, 10, 2, [(8, 20), (20, 32)]),

    ([(0, 10), (23, 33)], 40, 10, 2, [(0, 12), (22, 35)]),

    ([(5, 15), (25, 35)], 40, 10, 2, [(3, 15), (25, 37)]),
])
def test_apply_elos(blocks, n, m, e, want):
    assert apply_elos(blocks, n, m, e) == want

def test_elos_run_widens_zone_and_conserves():
    df = _strip([0, 0, 0, 3, 3, 0, 0, 0])
    res = run_dilution(_params(step_x=5.0, elos=5.0), df=df)
    ob = res.ore_blocks
    assert len(ob) == 1
    assert (ob.X_FROM.iloc[0], ob.X_TO.iloc[0]) == (10.0, 30.0)
    assert ob.AU_MII_DESIGN.iloc[0] == pytest.approx(3.0)
    assert ob.ELOS_TONNES.iloc[0] == pytest.approx(2 * 5 * 5 * 5 * 2.0)
    dm = res.diluted_model
    assert (dm.XINC * dm.YINC * dm.ZINC).sum() == pytest.approx(8 * 125)

def test_elos_must_be_whole_steps():
    with pytest.raises(ValueError, match="whole number of X steps"):
        _params(step_x=2.0, elos=3.0).validate()

def test_recover_keeps_above_cutoff_smu_beside_zone():

    df = _strip([0, 3, 3, 0, 1.2, 1.2, 0, 0])
    value = run_dilution(_params(ore_rule="value"), df=df).ore_blocks
    rec = run_dilution(_params(ore_rule="recover"), df=df).ore_blocks
    assert value.TONNES.sum() == pytest.approx(2 * 125 * 2.0)
    assert rec.TONNES.sum() == pytest.approx(5 * 125 * 2.0)
    assert (rec.AU_MII >= 1.0 - 1e-9).all()
    assert rec.XINC.min() >= 10.0

def test_recover_never_takes_a_block_below_cutoff():
    rng = np.random.default_rng(7)
    df = _strip(rng.uniform(0, 2.2, 60).round(2))
    ob = run_dilution(_params(ore_rule="recover"), df=df).ore_blocks
    assert len(ob)
    assert (ob.AU_MII >= 1.0 - 1e-9).all()

def test_trimmed_diluted_model_keeps_own_and_diluted_fields():
    df = _strip([0, 3, 3, 0, 0, 0])
    df["LITH"] = 7
    dm = run_dilution(_params(), df=df).diluted_model
    assert list(dm.columns) == ["MATERIAL", "XC", "YC", "ZC", "XINC", "YINC",
                                "ZINC", "AU", "D_AU", "RESCAT", "D_RESCAT",
                                "DENSITY"]
    ore = dm[dm.MATERIAL == "ORE"]
    assert ore.D_AU.iloc[0] == pytest.approx(3.0)

def test_rotated_outline_is_a_rotated_rectangle():
    r = pd.Series(dict(AZIMUTH=0.0, XC=0.0, YC=0.0, XINC=10.0, YINC=4.0, ZC=1.0,
                       X_FROM=0.0, X_TO=10.0))
    pts = block_outline(r)
    assert pts[0] == pts[-1]
    xs = sorted({round(p[0], 6) for p in pts})
    ys = sorted({round(p[1], 6) for p in pts})
    assert xs == [-2.0, 2.0] and ys == [-5.0, 5.0]

def test_region_over_whole_model_matches_no_regions():
    df = _grid_model()
    p = _params(cutoff=1.0, smu_x=10.0, step_x=1.0)
    base = run_dilution(p, df=df)
    whole = _rect("ALL", 60, 30, 120, 60, 90)
    res = run_dilution(_params(cutoff=1.0, smu_x=10.0, step_x=1.0,
                               regions=[whole]), df=df)
    pd.testing.assert_frame_equal(base.summary, res.summary)
    assert res.by_region.REGION.tolist() == ["ALL"]

def test_rotated_overlapping_regions_conserve_and_prioritise(tmp_path):
    df = _grid_model()
    regions = [_rect("NE", 40, 30, 60, 50, 120, 1), _rect("EW", 75, 30, 60, 60, 90, 2)]
    res = run_dilution(_params(cutoff=1.0, smu_x=10.0, step_x=1.0, step_y=1.0,
                               regions=regions, elos=1.0), df=df)
    dm = res.diluted_model
    vol = dm.XINC * dm.YINC * dm.ZINC
    assert vol.sum() == pytest.approx(len(df) * 125)
    assert (vol * dm.DENSITY * dm.D_AU).sum() == pytest.approx(
        (df.AU * df.DENSITY * 125).sum())
    assert not dm.duplicated(["XC", "YC", "ZC"]).any()
    ob = res.ore_blocks
    assert set(ob.REGION) <= {"NE", "EW", "MODEL"}
    assert {"NE", "EW"} <= set(ob.REGION)

    assert np.allclose(ob.loc[ob.REGION == "NE", "AZIMUTH"], 120.0)
    br = res.by_region
    assert br.DILUTED_TONNES.sum() == pytest.approx(ob.TONNES.sum())
    assert br.ORE_BLOCKS.sum() == len(ob)
    assert 0 <= res.ore_frac.min() and res.ore_frac.max() <= 1 + 1e-9
    paths = write_outputs(res, str(tmp_path))
    assert os.path.getsize(paths["by_region"]) > 0
    assert "Region      : NE priority 1" in open(paths["report"]).read()

def test_outside_waste_leaves_material_outside_regions_as_waste():
    df = _grid_model()
    region = _rect("W", 30, 30, 60, 60, 90)
    res = run_dilution(_params(cutoff=1.0, smu_x=10.0, step_x=1.0,
                               regions=[region], outside="waste"), df=df)
    ob = res.ore_blocks
    assert set(ob.REGION) == {"W"}
    assert (ob.X_TO <= 60 + 1e-9).all()

def test_threads_match_single_thread_with_regions():
    df = _grid_model(nx=16, ny=8, nz=3)
    regions = [_rect("R", 40, 20, 50, 40, 110, 1)]
    one = run_dilution(_params(cutoff=1.0, step_x=1.0, regions=regions), df=df)
    two = run_dilution(_params(cutoff=1.0, step_x=1.0, regions=regions, threads=2),
                       df=df)
    pd.testing.assert_frame_equal(one.ore_blocks, two.ore_blocks)
    pd.testing.assert_frame_equal(one.diluted_model, two.diluted_model)

def test_region_preview(tmp_path):
    pytest.importorskip("reportlab")
    from dilution.preview import plan_grid, region_preview, region_table
    df = _grid_model()
    regions = [_rect("NE", 40, 30, 60, 50, 120, 1), _rect("EW", 75, 30, 60, 60, 90, 2)]
    p = _params(cutoff=1.0, regions=regions)
    plan = plan_grid(df, p)
    assert plan["tonnes"].sum() == pytest.approx(len(df) * 125 * 2.6)
    above = (df.AU >= 1.0) & (df.RESCAT != 4)
    assert plan["ore"].sum() == pytest.approx(above.sum() * 125 * 2.6)
    tab = region_table(plan, order_regions(as_regions(regions)), p.outside)
    assert tab.ORE_TONNES.sum() == pytest.approx(plan["ore"].sum())
    path, _ = region_preview(df, p, str(tmp_path / "preview.pdf"))
    with open(path, "rb") as f:
        assert f.read(5) == b"%PDF-"

@pytest.mark.parametrize("audit", [False, True])
def test_diluted_model_dm_export_matches_csv(tmp_path, audit):
    from dilution.io_dm import read_dm
    df = _grid_model(nx=12, ny=6, nz=2)
    res = run_dilution(_params(cutoff=1.0, step_x=1.0, audit_fields=audit,
                               regions=[_rect("R", 30, 15, 40, 30, 110)]), df=df)
    paths = write_outputs(res, str(tmp_path))
    dm = read_dm(paths["diluted_model_dm"])
    csv = pd.read_csv(paths["diluted_model"])
    assert len(dm) == len(csv)
    assert (np.diff(dm.IJK) >= 0).all()
    assert all(len(c) <= 8 for c in dm.columns)
    assert dm.XMORIG.iloc[0] == 0.0 and dm.NX.iloc[0] == 12
    m = dm.merge(csv, on=["XC", "YC", "ZC"], suffixes=("", "_csv"))
    assert len(m) == len(csv)
    grade = "D_AU" if "D_AU" in csv else "AU"
    assert np.allclose(m[grade], m[grade + "_csv"])
    assert (m.MATERIAL == m.MATERIAL_csv).all()
    i = np.floor(m.XC / 5).astype(int)
    j = np.floor(m.YC / 5).astype(int)
    k = np.floor(m.ZC / 5).astype(int)
    assert (m.IJK == (k * 6 + j) * 12 + i).all()
