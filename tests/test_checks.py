import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dilution.checks import check_inputs, check_outputs
from dilution.core import DilutionParams, run_dilution
from test_core import _subcelled_model

def _p(**kw):
    base = dict(input_path="mem", output_dir="out", cutoff=1.0, smu_x=5,
                smu_y=5, smu_z=5, step_x=2.5, xinc_col="XINC", yinc_col="YINC",
                zinc_col="ZINC", oxide_col=None)
    base.update(kw)
    return DilutionParams(**base)

def _status(checks, name):
    return next(c.status for c in checks if c.name == name)

def test_good_model_passes_inputs_and_outputs():
    df = _subcelled_model()
    p = _p()
    pre = check_inputs(df, p)
    assert not [c for c in pre if c.status == "FAIL"], pre
    post = check_outputs(df, p, run_dilution(p, df=df))
    assert all(c.status == "PASS" for c in post), post

def test_unmapped_size_columns_fail():
    pre = check_inputs(_subcelled_model(), _p(xinc_col=None, yinc_col=None,
                                              zinc_col=None))
    assert _status(pre, "Sub-cell size columns") == "FAIL"

def test_bad_smu_fails():
    pre = check_inputs(_subcelled_model(), _p(smu_x=7.123, step_x=7.123))
    assert _status(pre, "SMU X vs parent") == "FAIL"

def test_off_grid_cells_fail():
    df = _subcelled_model()
    i = df.index[df.XINC == df.XINC.max()][3]
    df.loc[i, "XC"] += 3.0
    pre = check_inputs(df, _p())
    assert _status(pre, "Cells on the parent grid") == "FAIL"

def test_free_size_subcells_pass():
    df = _subcelled_model()
    whole = df.index[(df.XINC == 10) & (df.ZINC == 10)][:5]
    parts = []
    for i in whole:
        r = df.loc[i]
        for zlo, zhi in ((0, 3.7), (3.7, 10)):
            for xlo, xhi in ((0, 7.5), (7.5, 10)):
                q = r.copy()
                x0, z0 = r.XC - 5, r.ZC - 5
                q["XC"], q["XINC"] = x0 + (xlo + xhi) / 2, xhi - xlo
                q["ZC"], q["ZINC"] = z0 + (zlo + zhi) / 2, zhi - zlo
                parts.append(q)
    df = pd.concat([df.drop(index=whole), pd.DataFrame(parts)], ignore_index=True)
    p = _p(block_x=10, block_y=10, block_z=10)
    pre = check_inputs(df, p)
    assert not [c for c in pre if c.status == "FAIL"], pre
    post = check_outputs(df, p, run_dilution(p, df=df))
    assert all(c.status == "PASS" for c in post), post

def test_gap_and_bad_values_warn():
    df = _subcelled_model()
    df = df[~((df.XC > 1040) & (df.XC < 1050))].reset_index(drop=True)
    df.loc[0, "AU"] = -99
    df.loc[1, "RESCAT"] = 9
    pre = check_inputs(df, _p())
    assert _status(pre, "Gaps in X") == "WARN"
    assert _status(pre, "AU not negative") == "WARN"
    assert _status(pre, "RESCAT values") == "WARN"
