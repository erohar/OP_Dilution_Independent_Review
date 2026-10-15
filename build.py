from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(ROOT, "OpenPitDilution2D.spec")
APP_NAME = "OpenPitDilution2D"

log = logging.getLogger("build")

class BuildError(RuntimeError):
    pass

def exe_path() -> str:
    name = APP_NAME + (".exe" if sys.platform == "win32" else "")
    return os.path.join(ROOT, "dist", name)

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log.info("> %s", " ".join(cmd))
    try:
        return subprocess.run(cmd, cwd=ROOT, check=True, **kw)
    except FileNotFoundError as e:
        raise BuildError(f"Command not found: {cmd[0]}") from e
    except subprocess.CalledProcessError as e:
        out = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else ""
        raise BuildError(f"Failed with exit code {e.returncode}: {' '.join(cmd)}"
                         + (f"\n{out}" if out else "")) from e
    except subprocess.TimeoutExpired as e:
        raise BuildError(f"Timed out after {e.timeout:.0f} s: {' '.join(cmd)}") from e

def install() -> None:
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt",
         "-r", "requirements-build.txt"])

def run_tests() -> None:
    run([sys.executable, "-m", "pytest", "-q"])

def build() -> str:
    try:
        import PyInstaller
    except ImportError as e:
        raise BuildError("PyInstaller is not installed. Run "
                         "`python build.py --install` or "
                         "`pip install -r requirements-build.txt`.") from e
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", SPEC])
    path = exe_path()
    if not os.path.isfile(path):
        raise BuildError(f"PyInstaller finished but {path} is missing")
    log.info("Built %s (%.0f MB)", path, os.path.getsize(path) / 1e6)
    return path

SUBSYSTEM_WINDOWS_GUI = 2
SUBSYSTEM_WINDOWS_CUI = 3

def check_windowed(path: str) -> None:
    if not path.lower().endswith(".exe"):
        return
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        pe = int.from_bytes(head[0x3C:0x40], "little")
        if head[pe:pe + 4] != b"PE\0\0":
            raise BuildError(f"{path} is not a Windows executable")

        subsystem = int.from_bytes(head[pe + 92:pe + 94], "little")
    except OSError as e:
        raise BuildError(f"Could not read {path}: {e}") from e
    if subsystem != SUBSYSTEM_WINDOWS_GUI:
        kind = "a console" if subsystem == SUBSYSTEM_WINDOWS_CUI else f"subsystem {subsystem}"
        raise BuildError(f"{path} is {kind} program, so it would open a console "
                         "window; set console=False in the spec")
    log.info("Checked %s is a windowed program (no console window)", path)

def smoke_test(path: str) -> None:
    run([path, "--help"], capture_output=True, text=True, timeout=300)
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    try:
        from make_synthetic_model import make_model
    finally:
        sys.path.pop(0)
    work = tempfile.mkdtemp(prefix="dilution_smoke_")
    try:
        model = os.path.join(work, "model.csv")
        make_model(nx=40, ny=20, nz=4).to_csv(model, index=False)
        out = os.path.join(work, "out")
        t0 = time.time()
        run([path, model, "--out", out, "--cutoff", "0.5", "--smu-x", "5",
             "--step-x", "2.5", "--step-y", "5", "--oxide-col", "", "--threads", "2"],
            capture_output=True, text=True, timeout=900)
        made = sorted(os.listdir(out))
        if not any(f.startswith("diluted_model") for f in made):
            raise BuildError(f"Smoke run wrote no diluted model; got {made}")
        log.info("Smoke run passed in %.0f s: %s", time.time() - t0, ", ".join(made))
    finally:
        shutil.rmtree(work, ignore_errors=True)

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true",
                    help="pip install requirements.txt and requirements-build.txt first")
    ap.add_argument("--skip-tests", action="store_true", help="don't run pytest first")
    ap.add_argument("--skip-smoke", action="store_true",
                    help="don't run the built executable afterwards")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if a.install:
            install()
        if not a.skip_tests:
            run_tests()
        path = build()
        check_windowed(path)
        if not a.skip_smoke:
            smoke_test(path)
    except BuildError as e:
        log.error("%s", e)
        return 1
    except OSError as e:
        log.error("Build failed: %s", e)
        return 1
    log.info("Done. Share %s", exe_path())
    return 0

if __name__ == "__main__":
    sys.exit(main())
