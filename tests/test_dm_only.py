import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dilution import core
from dilution.core import iter_diluted_dm, run_dilution, write_outputs
from dilution.extras import run_sensitivity
from dilution.extras import _saved
from test_core import _params, _subcelled_model
from test_extras import _grid_model

KEY = ["ZC", "YC", "XC"]

def _sorted(f: pd.DataFrame) -> pd.DataFrame:
    return f.sort_values(KEY, kind="mergesort").reset_index(drop=True)

def _read_back(path, p) -> pd.DataFrame:
    return pd.concat(list(iter_diluted_dm(path, p, rows=50)), ignore_index=True)

def test_default_writes_csv_and_dm(tmp_path):
    res = run_dilution(_params(smu_x=10.0, step_x=5.0), df=_grid_model())
    paths = write_outputs(res, str(tmp_path))
    assert os.path.isfile(paths["diluted_model"])
    assert os.path.isfile(paths["diluted_model_dm"])

@pytest.mark.parametrize("model,kw", [
    (_grid_model, dict(smu_x=10.0, step_x=5.0)),
    (_grid_model, dict(smu_x=10.0, step_x=5.0, audit_fields=True)),
    (_subcelled_model, dict(smu_x=10.0, step_x=2.5, xinc_col="XINC",
                            yinc_col="YINC", zinc_col="ZINC")),
])
def test_dm_only_skips_csv_and_dm_holds_the_run_data(tmp_path, model, kw):
    p = _params(**kw)
    res = run_dilution(p, df=model())
    full = write_outputs(res, str(tmp_path / "both"))
    only = write_outputs(res, str(tmp_path / "dm"), diluted_csv=False)
    assert "diluted_model" not in only
    assert set(os.listdir(tmp_path / "dm")) == (
        set(os.listdir(tmp_path / "both")) - {"diluted_model_MII.csv"})

    with open(full["diluted_model_dm"], "rb") as a, open(only["diluted_model_dm"], "rb") as b:
        assert a.read() == b.read()

    csv = _sorted(pd.read_csv(full["diluted_model"]))
    dm = _sorted(_read_back(only["diluted_model_dm"], p))
    names = core._dm_names(list(csv.columns), p)
    csv = csv.rename(columns={c: n for c, n in names.items()
                              if n not in ("XC", "YC", "ZC", "XINC", "YINC", "ZINC")})
    assert set(dm.columns) == set(csv.columns) - {"PARENT_XINC", "PARENT_YINC",
                                                  "PARENT_ZINC"}
    pd.testing.assert_frame_equal(dm, csv[list(dm.columns)], check_dtype=False)

def test_dm_only_falls_back_to_csv_when_dm_fails(tmp_path, monkeypatch):
    res = run_dilution(_params(smu_x=10.0, step_x=5.0), df=_grid_model())

    def fail(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(core, "write_diluted_dm", fail)
    paths = write_outputs(res, str(tmp_path), diluted_csv=False)
    assert "diluted_model_dm" not in paths
    pd.testing.assert_frame_equal(pd.read_csv(paths["diluted_model"]),
                                  res.diluted_model, check_dtype=False)

def test_sensitivity_dm_only(tmp_path):
    df = _grid_model()
    p = _params(smu_x=10.0, step_x=5.0, output_dir=str(tmp_path))
    base = run_dilution(p, df=df)
    values = {"cutoff": [0.5, 1.0], "smu_x": [5.0]}
    both = run_sensitivity(p, df, values, base=base,
                           model_dir=str(tmp_path / "a" / "sensitivity_models"))
    only = run_sensitivity(p, df, values, base=base,
                           model_dir=str(tmp_path / "b" / "sensitivity_models"),
                           model_csv=False)
    files = os.listdir(tmp_path / "b" / "sensitivity_models")
    assert files and all(f.endswith(".dm") for f in files)
    for key in values:
        a, b = both[key], only[key]
        assert (b.MODEL == "").all() and (b.MODEL_DM != "").all()

        assert all(_saved(r) for r in b.to_dict("records"))
        pd.testing.assert_frame_equal(a.drop(columns=["MODEL"]),
                                      b.drop(columns=["MODEL"]))
        for v in values[key]:
            assert (a.attrs["curves"][v].curve([0.5, 1.0]).equals(
                b.attrs["curves"][v].curve([0.5, 1.0])))
