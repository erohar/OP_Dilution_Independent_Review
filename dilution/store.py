from __future__ import annotations

import atexit
import gc
import logging
import os
import shutil
import sys
import tempfile
import time
import weakref
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd

CHUNK_ROWS = 1_000_000
WORK_PREFIX = ".dilution_tmp_"
LEGACY_AGE = 3600.0

log = logging.getLogger(__name__)
_OPEN: "weakref.WeakSet" = weakref.WeakSet()

def _ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def _no_dmx(path: str) -> None:
    if _ext(path) == ".dmx":
        raise ValueError(".dmx files use Datamine's compressed format, which is "
                         "not publicly documented. In Studio, save the model as "
                         ".dm or export it to CSV, then load that file")

def read_header(path: str, skip_rows: int = 0) -> list[str]:
    _no_dmx(path)
    if _ext(path) == ".dm":
        from .io_dm import read_dm_header
        return read_dm_header(path)
    return [c.strip() for c in
            pd.read_csv(path, nrows=0, skiprows=skip_rows).columns]

def iter_chunks(path: str, skip_rows: int = 0, columns: Optional[list] = None,
                rows: int = CHUNK_ROWS) -> Iterator[pd.DataFrame]:
    _no_dmx(path)
    if _ext(path) == ".dm":
        from .io_dm import iter_dm
        yield from iter_dm(path, rows=rows, columns=columns)
        return
    wanted = set(columns) if columns is not None else None
    reader = pd.read_csv(path, skiprows=skip_rows, chunksize=rows,
                         usecols=(lambda c: str(c).strip() in wanted)
                         if wanted is not None else None)
    for ch in reader:
        ch.columns = [str(c).strip() for c in ch.columns]
        yield ch

def numeric_columns(p) -> list[str]:
    cols = [p.x_col, p.y_col, p.z_col, p.grade_col, p.rescat_col, p.density_col,
            p.xinc_col, p.yinc_col, p.zinc_col, *p.extra_numeric]
    return list(dict.fromkeys(c for c in cols if c))

def code_columns(p) -> list[str]:
    cols = [p.oxide_col, *p.extra_categorical, *p.extra_metal]
    if getattr(p, "dilute_attr_col", None):
        cols.append(p.dilute_attr_col)
    return list(dict.fromkeys(c for c in cols if c))

def _sorted_values(values: list) -> list[int]:
    idx = list(range(len(values)))
    try:
        return sorted(idx, key=lambda i: values[i])
    except TypeError:
        return sorted(idx, key=lambda i: str(values[i]))

def _work_root(p) -> Optional[str]:
    try:
        os.makedirs(p.output_dir, exist_ok=True)
        return p.output_dir
    except (OSError, TypeError):
        return None

def make_work_dir(root: Optional[str]) -> str:
    return tempfile.mkdtemp(prefix=f"{WORK_PREFIX}p{os.getpid()}_", dir=root)

def _owner(folder: str) -> Optional[int]:
    rest = os.path.basename(folder)[len(WORK_PREFIX):]
    head, sep, _ = rest.partition("_")
    if sep and head[:1] == "p" and head[1:].isdigit():
        return int(head[1:])
    return None

def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)
        if not h:
            return k32.GetLastError() == 5
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True

def _newest_mtime(folder: str) -> float:
    newest = os.path.getmtime(folder)
    for base, _, files in os.walk(folder):
        for name in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(base, name)))
            except OSError:
                pass
    return newest

def _is_stale(folder: str) -> bool:
    pid = _owner(folder)
    if pid is not None:
        return not _pid_alive(pid)
    try:
        return time.time() - _newest_mtime(folder) > LEGACY_AGE
    except OSError:
        return False

def sweep_work_dirs(root: Optional[str]) -> int:
    root = root or tempfile.gettempdir()
    try:
        names = [n for n in os.listdir(root) if n.startswith(WORK_PREFIX)]
    except OSError:
        return 0
    removed = 0
    for name in names:
        folder = os.path.join(root, name)
        try:
            if not os.path.isdir(folder) or not _is_stale(folder):
                continue
            shutil.rmtree(folder)
            removed += 1
        except OSError as e:
            log.debug("Could not delete the old work folder %s: %s", folder, e)
    if removed:
        log.info("Deleted %d temporary folder(s) left by earlier runs", removed)
    return removed

def _remove_work(num: dict, codes: dict, zindex: dict, work_dir: str) -> None:
    num.clear()
    codes.clear()
    zindex.clear()
    shutil.rmtree(work_dir, True)

@atexit.register
def _close_open_stores() -> None:
    for store in list(_OPEN):
        try:
            store.close()
        except Exception as e:
            log.debug("Could not delete work files at exit: %s", e)

class BlockStore:

    def __init__(self, columns: list, n: int, num: dict, codes: dict,
                 source=None, work_dir: Optional[str] = None):
        self.columns = list(columns)
        self.n = int(n)
        self._num = num
        self._codes = codes
        self.source = source
        self.work_dir = work_dir
        self._zindex: dict = {}
        self._finalizer = None
        if work_dir:

            self._finalizer = weakref.finalize(self, _remove_work, self._num,
                                               self._codes, self._zindex, work_dir)
            self._finalizer.atexit = False
            _OPEN.add(self)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, p) -> "BlockStore":
        num = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(float)
               for c in numeric_columns(p) if c in df.columns}
        codes = {}
        for c in code_columns(p):
            if c in df.columns:
                k, uniq = pd.factorize(df[c], sort=True)
                codes[c] = (k, list(uniq))
        return cls(list(df.columns), len(df), num, codes, source=df)

    @classmethod
    def build(cls, p, progress: Optional[Callable[[str], None]] = None,
              rows: int = CHUNK_ROWS, work_dir: Optional[str] = None) -> "BlockStore":
        header = read_header(p.input_path, p.skip_rows)
        want_num = [c for c in numeric_columns(p) if c in header]
        want_code = [c for c in code_columns(p) if c in header]
        if work_dir is None:
            root = _work_root(p)
            sweep_work_dirs(root)
            work = make_work_dir(root)
        else:
            work = work_dir
        store = None
        try:
            fnum = {c: open(os.path.join(work, f"n{i}.f8"), "wb")
                    for i, c in enumerate(want_num)}
            fcode = {c: open(os.path.join(work, f"c{i}.i4"), "wb")
                     for i, c in enumerate(want_code)}
            lookup = {c: {} for c in want_code}
            n = 0
            try:
                for ch in iter_chunks(p.input_path, p.skip_rows,
                                      list(dict.fromkeys(want_num + want_code)), rows):
                    for c in want_num:
                        fnum[c].write(pd.to_numeric(ch[c], errors="coerce")
                                      .to_numpy(np.float64).tobytes())
                    for c in want_code:
                        k, uniq = pd.factorize(ch[c])
                        lk = lookup[c]
                        g = np.array([lk.setdefault(u, len(lk)) for u in uniq] + [-1],
                                     dtype=np.int32)
                        fcode[c].write(g[k].astype(np.int32).tobytes())
                    n += len(ch)
                    if progress:
                        progress(f"Read {n:,} blocks")
            finally:
                for f in (*fnum.values(), *fcode.values()):
                    f.close()

            def mm(name, dtype, mode="r"):
                if n == 0:
                    return np.zeros(0, dtype=dtype)
                return np.memmap(os.path.join(work, name), dtype=dtype, mode=mode,
                                 shape=(n,))

            num = {c: mm(f"n{i}.f8", np.float64) for i, c in enumerate(want_num)}
            codes = {}
            for i, c in enumerate(want_code):
                values = list(lookup[c].keys())
                order = _sorted_values(values)
                rank = np.empty(len(values) + 1, dtype=np.int32)
                rank[order] = np.arange(len(values), dtype=np.int32)
                rank[-1] = -1
                arr = mm(f"c{i}.i4", np.int32, "r+")
                for s in range(0, n, rows):
                    arr[s:s + rows] = rank[arr[s:s + rows]]
                if n:
                    arr.flush()
                codes[c] = (mm(f"c{i}.i4", np.int32), [values[j] for j in order])
            store = cls(header, n, num, codes, source=p.input_path, work_dir=work)
            return store
        finally:
            if store is None and not work_dir:
                shutil.rmtree(work, ignore_errors=True)

    def has(self, name: Optional[str]) -> bool:
        return bool(name) and (name in self._num or name in self._codes)

    def num(self, name: str) -> np.ndarray:
        return self._num[name]

    def codes(self, name: str):
        return self._codes[name]

    def chunks(self, rows: int = CHUNK_ROWS) -> Iterator[slice]:
        for s in range(0, self.n, rows):
            yield slice(s, min(s + rows, self.n))

    def is_frame(self) -> bool:
        return isinstance(self.source, pd.DataFrame)

    def z_index(self, key, z_range: Callable[[slice], tuple]):
        if key not in self._zindex:
            zmin = np.empty(self.n)
            tall = 0.0
            for sl in self.chunks():
                lo, hi = z_range(sl)
                zmin[sl] = lo
                if len(lo):
                    tall = max(tall, float(np.nanmax(hi - lo)))
            order = np.argsort(zmin, kind="stable")
            self._zindex.clear()
            self._zindex[key] = (zmin[order], order, tall)
        return self._zindex[key]

    def close(self) -> None:
        self._num.clear()
        self._codes.clear()
        self._zindex.clear()

        gc.collect()
        info = self._finalizer.detach() if self._finalizer is not None else None
        if info is not None:

            _, func, args, _ = info
            func(*args)
            if os.path.isdir(self.work_dir):
                log.warning("Could not delete all temporary files in %s; a "
                            "later run deletes them", self.work_dir)
        _OPEN.discard(self)

def as_store(src, p, progress=None) -> BlockStore:
    if isinstance(src, BlockStore):
        return src
    if isinstance(src, pd.DataFrame):
        return BlockStore.from_frame(src, p)
    return BlockStore.build(p, progress=progress)
