import gc
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dilution import store as store_mod
from dilution.core import run_dilution, write_outputs
from dilution.preview import plan_grid
from dilution.store import (WORK_PREFIX, BlockStore, _owner,
                            make_work_dir, sweep_work_dirs)
from test_store import _model, _p

def _work_dirs(root):
    return sorted(n for n in os.listdir(root) if n.startswith(WORK_PREFIX))

def _setup(tmp_path):
    _model().to_csv(tmp_path / "m.csv", index=False)
    p = _p(tmp_path)
    os.makedirs(p.output_dir, exist_ok=True)
    return p

def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid

def test_work_dir_records_its_owner(tmp_path):
    assert _owner(make_work_dir(str(tmp_path))) == os.getpid()
    assert _owner(str(tmp_path / (WORK_PREFIX + "abc_12"))) is None

def _left_by(root, pid):
    folder = os.path.join(str(root), f"{WORK_PREFIX}p{pid}_x")
    os.makedirs(folder)
    return folder

def test_sweep_deletes_only_ended_runs(tmp_path):
    live = make_work_dir(str(tmp_path))
    dead = _left_by(tmp_path, _dead_pid())
    open(os.path.join(dead, "n0.f8"), "wb").write(b"x" * 100)
    old = tmp_path / (WORK_PREFIX + "legacy_old")
    old.mkdir()
    (old / "n0.f8").write_bytes(b"x")
    t = time.time() - store_mod.LEGACY_AGE - 60
    os.utime(old / "n0.f8", (t, t))
    os.utime(old, (t, t))
    recent = tmp_path / (WORK_PREFIX + "legacy_recent")
    recent.mkdir()
    (recent / "n0.f8").write_bytes(b"x")
    other = tmp_path / "keep_me"
    other.mkdir()
    assert sweep_work_dirs(str(tmp_path)) == 2
    assert os.path.isdir(live) and recent.is_dir() and other.is_dir()
    assert not os.path.exists(dead) and not old.exists()

def test_close_deletes_work_files(tmp_path):
    p = _setup(tmp_path)
    store = BlockStore.build(p)
    res = run_dilution(p, store=store)
    assert _work_dirs(p.output_dir)
    write_outputs(res, p.output_dir)
    res.close()
    assert _work_dirs(p.output_dir) == []

def _exit_with_open_store(tmp_path, hold_array):
    code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
            "from test_store import _p\n"
            "from dilution.store import BlockStore\n"
            "import pathlib\n"
            "s = BlockStore.build(_p(pathlib.Path(%r)))\n"
            % (os.path.join(os.path.dirname(__file__), ".."),
               os.path.dirname(__file__), str(tmp_path)))
    if hold_array:
        code += "a = s.num('AU')\n"
    subprocess.run([sys.executable, "-c", code], check=True)

def test_store_open_at_exit_is_deleted(tmp_path):
    p = _setup(tmp_path)
    _exit_with_open_store(tmp_path, hold_array=False)
    assert _work_dirs(p.output_dir) == []

def test_files_mapped_at_exit_deleted_by_next_run(tmp_path):

    p = _setup(tmp_path)
    _exit_with_open_store(tmp_path, hold_array=True)
    store = BlockStore.build(p)
    try:
        assert _work_dirs(p.output_dir) == [os.path.basename(store.work_dir)]
    finally:
        store.close()
    assert _work_dirs(p.output_dir) == []

def test_store_without_references_deleted_on_collect(tmp_path):
    p = _setup(tmp_path)
    store = BlockStore.build(p)
    work = store.work_dir
    del store
    gc.collect()
    assert not os.path.exists(work)

def test_next_run_sweeps_folders_left_by_a_killed_run(tmp_path):
    p = _setup(tmp_path)
    left = _left_by(p.output_dir, _dead_pid())
    store = BlockStore.build(p)
    try:
        assert not os.path.exists(left)
        assert _work_dirs(p.output_dir) == [os.path.basename(store.work_dir)]
    finally:
        store.close()

def test_drop_model_keeps_preview_files(tmp_path):
    p = _setup(tmp_path)
    store = BlockStore.build(p)
    res = run_dilution(p, store=store)
    ore = res.ore_blocks
    zc = float(ore["ZC"].iloc[0]) if len(ore) else 0.0
    before = res.ore_spool.bench(zc)
    cells = res.bench_cells(zc)
    res.spool.drop_model()
    assert not [f for f in os.listdir(res.spool.dir) if f.startswith("dm_")]
    assert res.ore_spool.bench(zc).equals(before)
    assert np.array_equal(res.bench_cells(zc)["X"], cells["X"])
    res.close()
    assert _work_dirs(p.output_dir) == []

def test_plan_grid_from_file_leaves_no_files(tmp_path):
    p = _setup(tmp_path)
    plan = plan_grid(None, p)
    assert plan["tonnes"].sum() > 0
    assert _work_dirs(p.output_dir) == []

def test_results_unchanged_by_cleanup(tmp_path):
    p = _setup(tmp_path)
    a = run_dilution(p, store=BlockStore.build(p))
    s1 = a.summary.copy()
    a.close()
    b = run_dilution(p, store=BlockStore.build(p))
    assert s1.equals(b.summary)
    b.close()
