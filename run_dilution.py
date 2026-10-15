import argparse
import json
import logging
import os
import sys
import time

from dilution.core import (ELOS_AXES, ORE_RULES, OUTSIDE, DilutionParams, format_report,
                           run_dilution, worker_pool, write_outputs)
from dilution.regions import load_regions
from dilution.store import BlockStore
from dilution.extras import (SENS_MODEL_DIR, append_history, format_sensitivity,
                             parse_list, run_sensitivity, write_sensitivity)

log = logging.getLogger("run_dilution")
LOG_FORMAT = "%(levelname)s: %(message)s"

def build_parser(description=__doc__):
    ap = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="block model CSV or Datamine .dm")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--cutoff", type=float, required=True)
    ap.add_argument("--smu-x", type=float, required=True)
    ap.add_argument("--smu-y", type=float, help="default: parent cell")
    ap.add_argument("--smu-z", type=float, help="default: parent cell")
    ap.add_argument("--step-x", type=float, required=True,
                    help="increment stepped along X when testing ore")
    ap.add_argument("--step-y", type=float, required=True,
                    help="increment ore blocks grow by in Y, across strips; SMU "
                         "Y must be a whole number of Y steps (a Y step equal "
                         "to SMU Y turns growth off)")
    ap.add_argument("--mode", choices=["MI", "MII"], default="MII")
    ap.add_argument("--x-col", default="XC")
    ap.add_argument("--y-col", default="YC")
    ap.add_argument("--z-col", default="ZC")
    ap.add_argument("--grade-col", default="AU")
    ap.add_argument("--rescat-col", default="RESCAT")
    ap.add_argument("--density-col", default="DENSITY",
                    help="use '' for a constant density")
    ap.add_argument("--oxide-col", default="OXIDE",
                    help="material type column; use '' for none")
    ap.add_argument("--default-density", type=float, default=2.7)
    ap.add_argument("--xinc-col")
    ap.add_argument("--yinc-col")
    ap.add_argument("--zinc-col")
    ap.add_argument("--block-x", type=float)
    ap.add_argument("--block-y", type=float)
    ap.add_argument("--block-z", type=float)
    ap.add_argument("--origin-x", type=float, help="parent grid origin (default infer)")
    ap.add_argument("--origin-y", type=float)
    ap.add_argument("--origin-z", type=float)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--extra-numeric", default="",
                    help="comma-separated columns to carry as tonnage-weighted averages")
    ap.add_argument("--extra-categorical", default="",
                    help="comma-separated code columns to carry as dominant by mass")
    ap.add_argument("--extra-metal", default="",
                    help="comma-separated code columns to carry as dominant by "
                         "metal (RESCAT is always carried this way)")
    ap.add_argument("--sens-cutoffs", default="",
                    help="cut-off sensitivity: a list '0.4,0.5,0.6' or a range "
                         "start:stop:step, e.g. '0.3:0.8:0.1'")
    ap.add_argument("--sens-smu-x", default="",
                    help="SMU X sensitivity, e.g. '5,10' or '5:20:5'")
    ap.add_argument("--sens-smu-y", default="",
                    help="SMU Y sensitivity, e.g. '5,10' or '5:20:5'")
    ap.add_argument("--sens-smu-z", default="",
                    help="SMU Z sensitivity, e.g. '2.5,5,10'")
    ap.add_argument("--sens-step-x", default="",
                    help="X step sensitivity, e.g. '0.5,1,2.5'")
    ap.add_argument("--sens-step-y", default="",
                    help="Y step sensitivity, e.g. '0.5,1,2.5'")
    ap.add_argument("--ore-rule", choices=list(ORE_RULES), default="recover",
                    help="recover: keep above cut-off SMUs beside ore zones; "
                         "value: keep only what adds metal above cut-off")
    ap.add_argument("--dilute-attr-col", default="",
                    help="dilute only blocks whose dominant attribute matches; "
                         "column name (e.g. WEATH)")
    ap.add_argument("--dilute-attr-values", default="",
                    help="comma-separated attribute values to dilute (e.g. OX,TR)")
    ap.add_argument("--elos", type=float, default=0.0,
                    help="edge dilution per side (m): at the ends of each ore "
                         "zone in X and on block edges facing waste in Y; a "
                         "whole number of X steps and of Y steps")
    ap.add_argument("--elos-axes", choices=list(ELOS_AXES), default="xy",
                    help="axes edge dilution applies on: 'xy' (default) X and "
                         "Y, 'x' the legacy X only (no edge dilution or growth "
                         "in Y: blocks stay one SMU Y deep)")
    ap.add_argument("--regions",
                    help="region rectangles (CSV or Surpac .str), each diluted "
                         "along its own axis")
    ap.add_argument("--outside", choices=list(OUTSIDE), default="model",
                    help="material outside every region: 'model' dilutes it "
                         "along the model X axis, 'waste' leaves it as waste")
    ap.add_argument("--sample-size", type=float,
                    help="piece size (m) for sharing blocks between regions")
    ap.add_argument("--preview-regions", action="store_true",
                    help="write regions_preview.pdf and stop, without running")
    ap.add_argument("--audit-fields", action="store_true",
                    help="keep the working columns (tonnes, metal, ids, tonnes "
                         "by RESCAT) in the diluted model")
    ap.add_argument("--merge-subcells", action="store_true",
                    help="merge sub-cells of the diluted model that lie in one "
                         "parent cell and have identical attributes, where "
                         "their union is a box (tonnes and metal are unchanged)")
    ap.add_argument("--merge-combine", default="",
                    help="with --merge-subcells: NAME=RULE,... for initial "
                         "attributes that don't have to be identical to merge "
                         "(RULE: avg_mass, avg_volume, avg_metal, dom_mass, "
                         "dom_volume or dom_metal; codes take dom_ only). "
                         "Attributes not listed must be identical; diluted "
                         "attributes always must be")
    ap.add_argument("--dm-only", action="store_true",
                    help="write the diluted models (main run and sensitivity "
                         "runs) only as Datamine .dm, without the CSV copy, "
                         "to save disk")
    ap.add_argument("--threads", default="auto",
                    help="worker threads, or 'auto' (default) to use the "
                         "recommended number for this computer and model")
    return ap

def params_from_args(a) -> DilutionParams:
    return DilutionParams(
        input_path=a.input, output_dir=a.out, x_col=a.x_col, y_col=a.y_col,
        z_col=a.z_col, grade_col=a.grade_col, rescat_col=a.rescat_col,
        density_col=a.density_col or None,
        oxide_col=a.oxide_col or None, default_density=a.default_density,
        xinc_col=a.xinc_col, yinc_col=a.yinc_col, zinc_col=a.zinc_col,
        block_x=a.block_x, block_y=a.block_y, block_z=a.block_z,
        origin_x=a.origin_x, origin_y=a.origin_y, origin_z=a.origin_z,
        cutoff=a.cutoff, smu_x=a.smu_x, smu_y=a.smu_y, smu_z=a.smu_z,
        step_x=a.step_x, step_y=a.step_y, mode=a.mode, skip_rows=a.skip_rows,
        threads=_threads(a.threads, a.input),
        extra_numeric=_names(getattr(a, "extra_numeric", "")),
        extra_categorical=_names(getattr(a, "extra_categorical", "")),
        extra_metal=_names(getattr(a, "extra_metal", "")),
        ore_rule=getattr(a, "ore_rule", "recover"),
        dilute_attr_col=getattr(a, "dilute_attr_col", "") or None,
        dilute_attr_values=_names(getattr(a, "dilute_attr_values", "")),
        elos=getattr(a, "elos", 0.0) or 0.0,
        elos_axes=getattr(a, "elos_axes", "xy"),
        regions=load_regions(a.regions) if getattr(a, "regions", None) else [],
        outside=getattr(a, "outside", "model"),
        sample_size=getattr(a, "sample_size", None),
        audit_fields=bool(getattr(a, "audit_fields", False)),
        merge_subcells=bool(getattr(a, "merge_subcells", False)),
        merge_rules=_rules(getattr(a, "merge_combine", "")))

def _rules(text) -> dict:
    out = {}
    for item in _names(text):
        name, sep, rule = item.partition("=")
        if not sep:
            raise ValueError(f"--merge-combine needs NAME=RULE, got {item!r}")
        out[name.strip()] = rule.strip()
    return out

def _names(text) -> list:
    return [t.strip() for t in (text or "").split(",") if t.strip()]

def _threads(value, path) -> int:
    if str(value).lower() != "auto":
        return max(1, int(value))
    from dilution.resources import detect, recommend_threads
    res = detect()
    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise FileNotFoundError(f"Block model not found: {path}") from e
    n = recommend_threads(res, size)
    log.info("Detected %s; using %d threads", res.describe(), n)
    return n

def attach_console() -> None:
    if sys.stdout is not None and sys.stderr is not None:
        return
    attached = False
    if sys.platform == "win32":
        try:
            import ctypes
            ATTACH_PARENT_PROCESS = -1
            attached = bool(ctypes.windll.kernel32.AttachConsole(ATTACH_PARENT_PROCESS))
        except (AttributeError, OSError) as e:
            log.debug("Could not attach to the parent console: %s", e)
    target = "CONOUT$" if attached else os.devnull
    try:
        if sys.stdout is None:
            sys.stdout = open(target, "w", encoding="utf-8", errors="replace")
        if sys.stderr is None:
            sys.stderr = open(target, "w", encoding="utf-8", errors="replace")
    except OSError as e:
        log.debug("Could not open %s for output: %s", target, e)

def close_splash() -> None:
    os.environ["PYINSTALLER_SUPPRESS_SPLASH_SCREEN"] = "1"
    try:
        import pyi_splash
    except ImportError:
        return
    try:
        if pyi_splash.is_alive():
            pyi_splash.close()
    except (RuntimeError, OSError) as e:
        log.debug("Could not close the splash screen: %s", e)

def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(level=level, format=LOG_FORMAT)

def sensitivity_lists(a) -> dict:
    return {k: parse_list(v) for k, v in (("cutoff", a.sens_cutoffs),
                                          ("smu_x", a.sens_smu_x),
                                          ("smu_y", a.sens_smu_y),
                                          ("smu_z", a.sens_smu_z),
                                          ("step_x", a.sens_step_x),
                                          ("step_y", a.sens_step_y)) if v}

def run_batch(a) -> dict:
    p = params_from_args(a)
    if not os.path.isfile(p.input_path):
        raise FileNotFoundError(f"Block model not found: {p.input_path}")
    sens = sensitivity_lists(a)
    t0 = time.time()

    store = BlockStore.build(p, progress=log.info)
    try:
        if getattr(a, "preview_regions", False):
            if not p.regions:
                raise ValueError("--preview-regions needs --regions")
            from dilution.preview import region_preview
            os.makedirs(p.output_dir, exist_ok=True)
            path, table = region_preview(store, p, os.path.join(
                p.output_dir, "regions_preview.pdf"), progress=log.info)
            log.info("\n%s", table.to_string(index=False))
            log.info("Region preview: %s", path)
            return {"regions_preview": path}

        with worker_pool(p.threads) as pool:
            res = run_dilution(p, store=store, pool=pool)
            dm_only = bool(getattr(a, "dm_only", False))
            paths = write_outputs(res, p.output_dir, diluted_csv=not dm_only)
            append_history(res, time.time() - t0)
            log.info("\n%s", format_report(res))
            if sens:
                tables = run_sensitivity(
                    p, store, sens, progress=log.info, base=res, pool=pool,
                    model_dir=os.path.join(p.output_dir, SENS_MODEL_DIR),
                    model_csv=not dm_only)
                paths.update(write_sensitivity(tables, p.output_dir, p.mode,
                                               res.params))
                log.info("\n%s", format_sensitivity(tables))
    finally:

        store.close()
    log.info("Outputs:\n  %s", "\n  ".join(paths.values()))
    return paths

def load_run_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run_dilution.py load-run",
        description="Load a previous run from its settings file, without running "
                    "the dilution again: print its report and, with "
                    "--rebuild-sensitivity, write its sensitivity curves "
                    "and bench CSV again from the saved models.")
    ap.add_argument("settings", help="settings JSON saved by the user interface")
    ap.add_argument("--rebuild-sensitivity", action="store_true",
                    help="write the sensitivity files again from the saved models")
    ap.add_argument("--out", help="folder for the rebuilt files (default: the run's folder)")
    return ap

def run_load(a) -> dict:
    from dilution.previous import (load_run, output_folder, read_settings,
                                   rebuild_sensitivity)
    run = load_run(a.settings)
    report = os.path.join(run.folder, "report.txt")
    if os.path.isfile(report):
        with open(report) as f:
            log.info("\n%s", f.read())
    log.info("Loaded the run in %s: %d benches, diluted model %s", run.folder,
             len(run.by_bench), os.path.basename(run.model))
    paths = {}
    if a.rebuild_sensitivity:
        folder = output_folder(read_settings(a.settings), a.settings)
        paths = rebuild_sensitivity(folder, a.out)
        log.info("Outputs:\n  %s", "\n  ".join(paths.values()))
    return paths

def cli(argv) -> int:
    load = bool(argv) and argv[0] == "load-run"
    a = (load_run_parser() if load else build_parser()).parse_args(
        argv[1:] if load else argv)
    setup_logging()
    try:
        run_load(a) if load else run_batch(a)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    except PermissionError as e:
        log.error("Cannot read or write %s: %s", e.filename, e.strerror)
        return 1
    except (ValueError, KeyError, json.JSONDecodeError) as e:

        log.error("%s", e)
        return 1
    except MemoryError:
        log.error("Ran out of memory. Use fewer threads or free some RAM.")
        return 1
    except Exception:
        log.exception("The dilution run failed")
        return 1
    return 0

def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    attach_console()
    if argv:
        close_splash()
        return cli(argv)

    setup_logging(logging.WARNING)
    try:
        from dilution.gui import main as gui_main
    except ImportError as e:
        close_splash()
        log.error("The user interface needs Tkinter (%s). Pass arguments for a "
                  "batch run instead; see --help.", e)
        return 1

    gui_main(on_ready=close_splash)
    return 0

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
