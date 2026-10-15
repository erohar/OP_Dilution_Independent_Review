import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dilution.checks import check_inputs, check_outputs, format_checks
from dilution.core import run_dilution, write_outputs
from dilution.io_dm import iter_dm, read_dm, write_dm
from dilution.store import BlockStore
from test_core import _params, _subcelled_model

def _model():
    df = _subcelled_model()
    rng = np.random.default_rng(3)
    df["OXIDE"] = rng.choice(["TR", "OX", "FR"], len(df))
    df["AG"] = rng.lognormal(0, 1, len(df))
    df["LITH"] = rng.choice([3, 1, 2], len(df))
    return df

def _p(tmp_path, **kw):
    base = dict(input_path=str(tmp_path / "m.csv"), output_dir=str(tmp_path / "out"),
                block_x=None, block_y=None, block_z=None, smu_x=5, smu_y=5,
                smu_z=5, step_x=2.5, xinc_col="XINC", yinc_col="YINC",
                zinc_col="ZINC", oxide_col="OXIDE", extra_numeric=["AG"],
                extra_categorical=["LITH"], extra_metal=["OXIDE"])
    base.update(kw)
    return _params(**base)

def _same(a, b):
    kw = dict(check_dtype=False, check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(a.ore_blocks, b.ore_blocks, **kw)
    pd.testing.assert_frame_equal(a.diluted_model, b.diluted_model, **kw)
    assert np.allclose(a.ore_frac, b.ore_frac)
    pd.testing.assert_frame_equal(a.summary, b.summary, **kw)

def test_store_from_file_in_chunks_matches_frame(tmp_path):
    df = _model()
    df.to_csv(tmp_path / "m.csv", index=False)
    p = _p(tmp_path)
    store = BlockStore.build(p, rows=7)
    assert store.n == len(df) and store.work_dir
    codes, values = store.codes("OXIDE")
    assert values == ["FR", "OX", "TR"]
    assert [values[c] for c in codes] == list(df.OXIDE)
    assert np.allclose(store.num("AU"), df.AU.to_numpy(), rtol=1e-14, atol=0)
    _same(run_dilution(p, df=df), run_dilution(p, store=store))
    work = store.work_dir
    del codes
    store.close()
    assert not os.path.exists(work)

def test_split_jobs_match_whole_benches(tmp_path):
    df = _model()
    p = _p(tmp_path)
    whole = run_dilution(p, df=df)
    split = run_dilution(p, df=df, job_cells=10)
    assert split.stats["jobs"] > whole.stats["jobs"]
    pd.testing.assert_frame_equal(whole.ore_blocks, split.ore_blocks,
                                  check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(whole.diluted_model, split.diluted_model,
                                  check_exact=False, rtol=1e-9)
    assert np.allclose(whole.ore_frac, split.ore_frac)

def test_split_jobs_on_threads_with_smu_crossing_parents(tmp_path):
    df = _model()
    p = _p(tmp_path, smu_x=7.5, smu_y=7.5, smu_z=7.5, threads=2)
    one = run_dilution(_p(tmp_path, smu_x=7.5, smu_y=7.5, smu_z=7.5), df=df)
    two = run_dilution(p, df=df, job_cells=10)
    pd.testing.assert_frame_equal(one.ore_blocks, two.ore_blocks,
                                  check_exact=False, rtol=1e-9)
    pd.testing.assert_frame_equal(one.diluted_model, two.diluted_model,
                                  check_exact=False, rtol=1e-9)

def test_outputs_streamed_to_disk(tmp_path):
    df = _model()
    df.to_csv(tmp_path / "m.csv", index=False)
    p = _p(tmp_path)
    store = BlockStore.build(p, rows=11)
    res = run_dilution(p, store=store, job_cells=10)
    paths = write_outputs(res, p.output_dir)
    dm = pd.read_csv(paths["diluted_model"])
    assert dm.TONNES.sum() == pytest.approx((df.XINC * df.YINC * df.ZINC * df.DENSITY).sum())
    assert (dm.ZC.diff().dropna() <= 0).all()
    bm = pd.read_csv(paths["block_model"])
    assert len(bm) == len(df) and list(bm.columns[:len(df.columns)]) == list(df.columns)
    assert np.allclose(bm.ORE_FRAC, res.ore_frac.clip(0, 1))
    assert not [c for c in check_outputs(store, p, res) if c.status != "PASS"]

def test_checks_same_from_store_and_frame(tmp_path):
    df = _model()
    df = df[~((df.XC > 1040) & (df.XC < 1050))].reset_index(drop=True)
    df.to_csv(tmp_path / "m.csv", index=False)
    p = _p(tmp_path)
    a = format_checks(check_inputs(df, p))
    b = format_checks(check_inputs(BlockStore.build(p, rows=13), p))
    assert a == b
    assert "WARN   Gaps in X" in a

def test_dm_read_in_chunks(tmp_path):
    n = 1000
    df = pd.DataFrame({"A": np.arange(n, dtype=float), "T": ["ab", "cd"] * (n // 2)})
    path = str(tmp_path / "m.dm")
    write_dm(df, path)
    parts = list(iter_dm(path, rows=100))
    assert len(parts) == 4
    out = pd.concat(parts, ignore_index=True)
    pd.testing.assert_frame_equal(out, read_dm(path))
    assert list(out["T"][:4]) == ["ab", "cd", "ab", "cd"]
    only = next(iter_dm(path, rows=None, columns=["T"]))
    assert list(only.columns) == ["T"] and len(only) == n
