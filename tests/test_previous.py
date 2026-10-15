import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_dilution
from dilution.core import run_dilution as run, write_outputs
from dilution.extras import run_sensitivity, write_sensitivity
from dilution.previous import (LoadedRun, load_run, output_folder,
                               read_settings, rebuild_sensitivity)
from test_core import _params
from test_extras import _grid_model

def _saved_run(tmp_path, dm_only=True, sens=True):
    out = str(tmp_path / "run")
    p = _params(smu_x=10.0, step_x=5.0, output_dir=out)
    df = _grid_model()
    res = run(p, df=df)
    write_outputs(res, out, diluted_csv=not dm_only)
    if sens:
        tables = run_sensitivity(p, df, {"cutoff": [0.5, 1.0]}, base=res,
                                 model_dir=os.path.join(out, "sensitivity_models"),
                                 model_csv=not dm_only)
        write_sensitivity(tables, out, p.mode, res.params)
    settings = str(tmp_path / "settings.json")
    with open(settings, "w") as f:
        json.dump({"output": out}, f)
    return res, out, settings

@pytest.mark.parametrize("dm_only", [True, False])
def test_loaded_run_matches_the_live_run(tmp_path, dm_only):
    res, out, settings = _saved_run(tmp_path, dm_only, sens=False)
    run_ = load_run(settings)
    assert run_.model.endswith(".dm")
    pd.testing.assert_frame_equal(run_.by_bench, res.by_bench.reset_index(drop=True),
                                  check_dtype=False)
    assert run_.grid["xhi"] - run_.grid["xlo"] == pytest.approx(
        res.params["parent_nx"] * res.params["block_x"])
    zc = float(res.by_bench.ZC.iloc[0])
    pd.testing.assert_frame_equal(run_.ore_spool.bench(zc).reset_index(drop=True),
                                  res.ore_spool.bench(zc).reset_index(drop=True),
                                  check_dtype=False)

    c = run_.bench_cells(zc)
    ore_area = float((c["ore"] * c["xs"] * c["ys"]).sum())
    ob = res.ore_spool.bench(zc)
    assert ore_area == pytest.approx(float((ob.XINC * ob.YINC).sum()))
    assert run_.bench_cells(zc) is c

def test_loaded_run_falls_back_to_the_csv(tmp_path):
    res, out, settings = _saved_run(tmp_path, dm_only=False, sens=False)
    os.remove(os.path.join(out, "diluted_model_MII.dm"))
    run_ = load_run(settings)
    assert run_.model.endswith(".csv")
    zc = float(res.by_bench.ZC.iloc[0])
    assert run_.bench_cells(zc)["ore"].sum() > 0

def test_missing_run_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="output folder"):
        output_folder({"output": str(tmp_path / "none")})
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="nothing to load"):
        LoadedRun(str(tmp_path / "empty"))
    bad = tmp_path / "bad.json"
    bad.write_text("[1]")
    with pytest.raises(ValueError, match="not a settings file"):
        read_settings(str(bad))
    bad.write_text("{")
    with pytest.raises(ValueError, match="not valid JSON"):
        read_settings(str(bad))

@pytest.mark.parametrize("dm_only", [True, False])
def test_rebuilt_sensitivity_matches_the_run(tmp_path, dm_only):
    res, out, settings = _saved_run(tmp_path, dm_only)
    mode = res.params["mode"]
    names = [f"sensitivity_curves_{mode}.csv", f"sensitivity_benches_{mode}.csv"]
    before = {n: pd.read_csv(os.path.join(out, n)) for n in names}
    again = str(tmp_path / "again")
    paths = rebuild_sensitivity(out, again)
    assert set(paths) == {"sensitivity_curves", "sensitivity_benches"}
    for n in names:
        pd.testing.assert_frame_equal(pd.read_csv(os.path.join(again, n)), before[n],
                                      check_dtype=False, rtol=1e-9, atol=1e-9)
    assert set(os.listdir(again)) == set(names)

def test_rebuild_needs_sensitivity_tables(tmp_path):
    res, out, settings = _saved_run(tmp_path, sens=False)
    with pytest.raises(ValueError, match="No sensitivity tables"):
        rebuild_sensitivity(out)

def test_cli_load_run(tmp_path):
    res, out, settings = _saved_run(tmp_path)
    again = str(tmp_path / "again")
    assert run_dilution.cli(["load-run", settings, "--rebuild-sensitivity",
                             "--out", again]) == 0
    assert os.path.isfile(os.path.join(again, "sensitivity_benches_MII.csv"))
    assert run_dilution.cli(["load-run", settings]) == 0
    assert run_dilution.cli(["load-run", str(tmp_path / "missing.json")]) == 1
