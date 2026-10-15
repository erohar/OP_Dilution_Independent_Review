import os
import sys
import threading

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dilution.core import (DilutionCancelled, load_block_model,
                           read_header, run_dilution)
from dilution.extras import (GradeTonnage, append_history, fit_step,
                             parse_list, read_history, run_sensitivity,
                             write_sensitivity)
from dilution.io_dm import read_dm, write_dm
from test_core import _params, _strip_model

@pytest.mark.parametrize("double", [False, True])
def test_dm_round_trip(tmp_path, double):
    df = pd.DataFrame({"XC": [2.5, 7.5, 12.5], "AU": [0.1, np.nan, 3.25],
                       "OXIDE": ["OX", "TRANS", ""], "RESCAT": [1.0, 2.0, 4.0]})
    path = str(tmp_path / "m.dm")
    write_dm(df, path, double=double, implicit={"XMORIG": 100.0})
    out = read_dm(path)
    assert list(out.columns) == ["XC", "AU", "OXIDE", "RESCAT", "XMORIG"]
    assert np.allclose(out.XC, df.XC)
    assert np.isnan(out.AU[1]) and out.AU[2] == pytest.approx(3.25)
    assert list(out.OXIDE) == ["OX", "TRANS", ""]
    assert (out.XMORIG == 100.0).all()

def test_dm_many_pages(tmp_path):
    n = 1000
    df = pd.DataFrame({"A": np.arange(n, dtype=float), "B": np.arange(n) * 2.0})
    path = str(tmp_path / "m.dm")
    write_dm(df, path)
    out = read_dm(path)
    assert len(out) == n and (out.B == df.B).all()

def test_dm_model_runs_like_csv(tmp_path):
    df = _strip_model([0, 0, 0, 5.0, 0, 0, 0, 0])
    write_dm(df, str(tmp_path / "m.dm"))
    df.to_csv(tmp_path / "m.csv", index=False)
    assert read_header(str(tmp_path / "m.dm")) == list(df.columns)
    a = run_dilution(_params(input_path=str(tmp_path / "m.dm"),
                             output_dir=str(tmp_path / "a")))
    b = run_dilution(_params(input_path=str(tmp_path / "m.csv"),
                             output_dir=str(tmp_path / "b")))
    pd.testing.assert_frame_equal(a.ore_blocks, b.ore_blocks, check_dtype=False)

def test_dmx_gives_clear_error(tmp_path):
    path = tmp_path / "m.dmx"
    path.write_bytes(b"\0" * 64)
    with pytest.raises(ValueError, match="save the model as .dm or export it to CSV"):
        read_header(str(path))
    with pytest.raises(ValueError, match=".dm"):
        load_block_model(_params(input_path=str(path)))

def test_extra_attributes_weighted_and_dominant():
    df = _strip_model([0, 3, 3, 3, 0])
    df["AG"] = [0.0, 10.0, 20.0, 30.0, 0.0]
    df["LITH"] = ["A", "B", "B", "C", "C"]
    df.loc[1, "DENSITY"] = 4.0
    ob = run_dilution(_params(smu_x=15.0, extra_numeric=["AG"],
                              extra_categorical=["LITH"]), df=df).ore_blocks
    assert len(ob) == 1
    assert ob.AG.iloc[0] == pytest.approx((2 * 10 + 20 + 30) / 4)
    assert ob.LITH.iloc[0] == "B"
    dm = run_dilution(_params(smu_x=15.0, grade_col="AU", extra_numeric=["AG"],
                              extra_categorical=["LITH"]), df=df).diluted_model
    ore = dm[dm.MATERIAL == "ORE"]
    assert (ore.D_AG == ob.AG.iloc[0]).all() and (ore.D_LITH == "B").all()
    assert "AG" not in dm.columns

def _grid_model():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(24):
        for j in range(4):
            rows.append(dict(XC=2.5 + 5 * i, YC=2.5 + 5 * j, ZC=2.5,
                             AU=float(rng.lognormal(-0.5, 1.0)), RESCAT=2, DENSITY=2.0))
    return pd.DataFrame(rows)

def test_sensitivity_runs_are_independent_runs():
    df = _grid_model()
    p = _params(smu_x=10.0, step_x=5.0)
    base = run_dilution(p, df=df)
    t = run_sensitivity(p, df, {"cutoff": [0.5, 1.0, 2.0], "smu_x": [5.0, 20.0],
                                "smu_y": [5.0, 10.0]}, base=base)
    assert set(t) == {"cutoff", "smu_x", "smu_y"}
    import dataclasses
    for key, table in t.items():
        for r in table.to_dict("records"):
            direct = run_dilution(dataclasses.replace(p, **{key: r["VALUE"]}), df=df)
            assert r["DILUTED_TONNES"] == pytest.approx(
                direct.summary.iloc[1].TONNES), (key, r)
    assert t["cutoff"].NOTE.tolist() == ["", "base case", ""]
    assert (t["smu_y"].SMU_Y == [5.0, 10.0]).all()

def test_sensitivity_smu_z_and_step_x():
    df = _grid_model()
    df2 = pd.concat([df, df.assign(ZC=7.5)], ignore_index=True)
    p = _params(smu_x=10.0, step_x=5.0)
    base = run_dilution(p, df=df2)
    t = run_sensitivity(p, df2, {"smu_z": [2.5, 10.0], "step_x": [2.5, 5.0, 4.0]},
                        base=base)
    import dataclasses
    for r in t["smu_z"].to_dict("records"):
        direct = run_dilution(dataclasses.replace(p, smu_z=r["VALUE"]), df=df2)
        assert r["DILUTED_TONNES"] == pytest.approx(direct.summary.iloc[1].TONNES)
        assert r["SMU_Z"] == r["VALUE"]
    st = t["step_x"]
    assert list(st.STEP_X[:2]) == [2.5, 5.0] and st.NOTE[1] == "base case"
    assert "whole number of X steps" in st.NOTE[2]

def test_sensitivity_smu_x_not_forced_to_step():
    df = _grid_model()

    t = run_sensitivity(_params(smu_x=10.0, step_x=5.0), df,
                        {"smu_x": [2.5, 7.5, 7.123]})
    ok, odd, bad = (t["smu_x"].iloc[i] for i in range(3))
    assert ok.NOTE == "" and ok.STEP_X == 2.5 and ok.DILUTED_TONNES > 0
    assert odd.NOTE == "" and odd.STEP_X == 2.5
    assert bad.NOTE != ""

def test_fit_step():
    assert fit_step(10, 1, 20) == 1
    assert fit_step(7, 0.5, 7) == 0.5
    assert fit_step(5.25, 0.5, 21) == 0.25
    assert fit_step(5, 0.3, 20) == pytest.approx(0.1)
    assert fit_step(2.5, 5, 5) == 2.5

def test_write_sensitivity_one_file_per_parameter(tmp_path):
    t = run_sensitivity(_params(), _grid_model(), {"cutoff": [0.5], "smu_y": [5.0]})
    paths = write_sensitivity(t, str(tmp_path), "MII")
    assert sorted(os.path.basename(v) for v in paths.values()) == [
        "sensitivity_cutoff_MII.csv", "sensitivity_smu_y_MII.csv"]

@pytest.mark.parametrize("audit", [True, False])
def test_sensitivity_saves_a_model_per_run(tmp_path, audit):
    df = _grid_model()
    p = _params(smu_x=10.0, step_x=5.0, audit_fields=audit, output_dir=str(tmp_path))
    base = run_dilution(p, df=df)
    mdir = str(tmp_path / "sensitivity_models")
    t = run_sensitivity(p, df, {"cutoff": [0.5, 1.0], "smu_x": [5.0, 7.123]},
                        base=base, model_dir=mdir)
    assert sorted(os.listdir(mdir)) == sorted(
        f"diluted_model_MII_{n}{ext}" for n in ("cutoff_0.5", "cutoff_1", "smu_x_5")
        for ext in (".csv", ".dm"))
    assert t["smu_x"].MODEL.isna().tolist() == [False, True]
    assert t["smu_x"].MODEL_DM.isna().tolist() == [False, True]
    import dataclasses
    for key, table in t.items():
        for r in table.to_dict("records"):
            if not isinstance(r["MODEL"], str):
                continue

            direct = run_dilution(dataclasses.replace(p, **{key: r["VALUE"]}), df=df)
            saved = pd.read_csv(tmp_path / r["MODEL"])
            pd.testing.assert_frame_equal(saved, direct.diluted_model,
                                          check_dtype=False)

            dm = read_dm(str(tmp_path / r["MODEL_DM"]))
            assert len(dm) == len(saved)
            grade = "D_AU" if "D_AU" in dm.columns else next(
                c for c in dm.columns if c.startswith("D_"))
            assert dm[grade].sum() == pytest.approx(saved["D_AU"].sum())

            c = t[key].attrs["curves"][r["VALUE"]].curve([direct.params["cutoff"]])
            assert c.TONNES[0] == pytest.approx(r["DILUTED_TONNES"])
            assert c.GRADE[0] == pytest.approx(r["DILUTED_AU"])

def test_grade_tonnage_bins():
    gt = GradeTonnage(step=0.5)
    gt.add(np.array([0.2, 1.0, 1.4, np.nan, -1.0]), [10.0, 20.0, 30.0, 5.0, 5.0])
    gt.add(np.array([3.0]), [40.0])
    c = gt.curve([0.0, 1.0, 1.5, 3.0, 9.0])
    assert c.TONNES.tolist() == [110.0, 90.0, 40.0, 40.0, 0.0]
    assert c.GRADE[1] == pytest.approx((20 + 42 + 120) / 90)
    assert c.GRADE[4] == 0.0

def test_write_sensitivity_curves_and_no_pdfs(tmp_path):
    df = _grid_model()
    p = _params(smu_x=10.0, step_x=5.0)
    base = run_dilution(p, df=df)
    t = run_sensitivity(p, df, {"cutoff": [0.5, 1.0, 2.0], "smu_y": [5.0, 10.0]},
                        base=base, model_dir=str(tmp_path / "sensitivity_models"))
    paths = write_sensitivity(t, str(tmp_path), "MII", base.params)
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".pdf")]
    curves = pd.read_csv(paths["sensitivity_curves"])
    assert set(curves.PARAMETER) == {"Cut-off", "SMU Y"}
    assert curves.groupby(["PARAMETER", "VALUE"]).size().nunique() == 1
    assert curves[curves.BASE].VALUE.tolist().count(1.0) > 0

def test_summary_only_matches_full_run():
    df = _grid_model()
    full = run_dilution(_params(), df=df)
    quick = run_dilution(_params(), df=df, summary_only=True)
    pd.testing.assert_frame_equal(full.ore_blocks, quick.ore_blocks)
    pd.testing.assert_frame_equal(full.summary, quick.summary)
    assert len(quick.diluted_model) == 0

def test_cancel_stops_run():
    ev = threading.Event()
    ev.set()
    with pytest.raises(DilutionCancelled):
        run_dilution(_params(), df=_strip_model([0, 5.0, 0, 0]), cancel=ev)

def test_history_appends(tmp_path):
    out = tmp_path / "out"
    res = run_dilution(_params(output_dir=str(out)),
                       df=_strip_model([0, 5.0, 0, 0, 0]))
    hist = str(tmp_path / "h" / "history.csv")
    append_history(res, 1.23, hist)
    append_history(res, 2.0, hist)
    h = read_history(hist)
    assert len(h) == 2 and list(h.SECONDS) == [1.2, 2.0]
    assert h.CUTOFF.iloc[0] == 1.0 and h.ORE_BLOCKS.iloc[0] == 1
    assert len(pd.read_csv(out / "run_history.csv")) == 2

def test_parse_list():
    assert parse_list("0.5, 0.8 1") == [0.5, 0.8, 1.0]
    assert parse_list("") == []
    assert parse_list("0.5\r\n0.6\r\n0.7\r\n") == [0.5, 0.6, 0.7]
    assert parse_list("5\t10\t15") == [5, 10, 15]
    assert parse_list("0.3:0.8:0.1") == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert parse_list("5 to 20 step 5") == [5, 10, 15, 20]
    with pytest.raises(ValueError):
        parse_list("1:2")

def test_mi_run_names_definition_grade_and_file(tmp_path):
    from dilution.core import write_outputs
    res = run_dilution(_params(mode="MI", output_dir=str(tmp_path)),
                       df=_strip_model([0, 5.0, 0, 0, 0]))
    assert "AU_MI" in res.ore_blocks.columns and "AU_MI" in res.block_model.columns
    assert {"D_AU", "AU_MI"} <= set(res.diluted_model.columns)
    paths = write_outputs(res, str(tmp_path))
    assert os.path.basename(paths["diluted_model"]) == "diluted_model_MI.csv"

def test_rescat_and_extra_metal_are_dominant_by_metal():
    df = _strip_model([0, 2, 6, 2, 0], [3, 3, 2, 3, 3])
    df["LITH"] = ["A", "B", "C", "B", "A"]
    res = run_dilution(_params(smu_x=15.0, extra_categorical=["LITH"],
                               extra_metal=["LITH"]), df=df)
    ob, dm = res.ore_blocks, res.diluted_model
    assert len(ob) == 1
    assert ob.LITH.iloc[0] == "B" and ob.LITH_METAL.iloc[0] == "C"

    assert ob.RESCAT.iloc[0] == 2
    assert ob.TONNES_RESCAT_3.iloc[0] > ob.TONNES_RESCAT_2.iloc[0]
    assert (dm[dm.MATERIAL == "ORE"].D_RESCAT == 2).all()

    assert set(dm[dm.MATERIAL == "WASTE"].D_RESCAT) == {3}

def test_extra_metal_vs_mass():
    df = _strip_model([0, 2, 6, 2, 0])
    df["LITH"] = ["A", "B", "C", "B", "A"]
    df["LITH2"] = df.LITH
    ob = run_dilution(_params(smu_x=15.0, extra_categorical=["LITH"],
                              extra_metal=["LITH2"]), df=df).ore_blocks
    assert ob.LITH.iloc[0] == "B" and ob.LITH2.iloc[0] == "C"

def test_sensitivity_step_y_and_smu_y_fit():
    df = _grid_model()
    p = _params(smu_x=10.0, step_x=5.0, smu_y=5.0, step_y=1.0)
    base = run_dilution(p, df=df)
    t = run_sensitivity(p, df, {"step_y": [1.0, 2.5, 5.0, 2.0],
                                "smu_y": [5.0, 7.5]}, base=base)
    st = t["step_y"]
    assert list(st.STEP_Y[:3]) == [1.0, 2.5, 5.0] and st.NOTE[0] == "base case"
    assert "whole number of Y steps" in st.NOTE[3]
    import dataclasses
    for r in st.to_dict("records")[1:3]:
        direct = run_dilution(dataclasses.replace(p, step_y=r["VALUE"]), df=df)
        assert r["DILUTED_TONNES"] == pytest.approx(direct.summary.iloc[1].TONNES)
    sm = t["smu_y"]
    assert list(sm.STEP_Y) == [1.0, 0.5] and list(sm.NOTE) == ["base case", ""]

def test_history_from_an_older_version_gets_the_new_columns(tmp_path):
    hist = tmp_path / "history.csv"
    pd.DataFrame([{"RUN_TIME": "old", "STEP_X": 1.0}]).to_csv(hist, index=False)
    res = run_dilution(_params(output_dir=str(tmp_path / "o")), df=_grid_model())
    os.makedirs(tmp_path / "o", exist_ok=True)
    append_history(res, 1.0, str(hist))
    h = read_history(str(hist))
    assert len(h) == 2 and "STEP_Y" in h.columns and h.RUN_TIME[0] == "old"
