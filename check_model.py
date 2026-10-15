import argparse
import json
import logging
import os
import sys

import pandas as pd

from dilution.checks import (check_inputs, check_outputs, format_checks,
                             params_from_settings)
from dilution.core import format_report, run_dilution, write_outputs
from dilution.store import BlockStore
from run_dilution import build_parser, params_from_args, setup_logging

log = logging.getLogger("check_model")

def load_params(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--settings")
    known, rest = pre.parse_known_args(argv)
    if not known.settings:
        return params_from_args(build_parser(__doc__).parse_args(rest))
    try:
        return params_from_settings(known.settings)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Settings file not found: {known.settings}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Settings file {known.settings} is not valid JSON: "
                         f"{e.msg} (line {e.lineno}, column {e.colno})") from e
    except KeyError as e:
        raise ValueError(f"Settings file {known.settings} has no {e} entry") from e

def run_checks(p) -> int:
    if not os.path.isfile(p.input_path):
        raise FileNotFoundError(f"Block model not found: {p.input_path}")

    store = BlockStore.build(p, progress=log.info)
    try:
        checks = check_inputs(store, p)
        log.info("Input model checks\n\n%s", format_checks(checks))
        if any(c.status == "FAIL" for c in checks):
            log.error("Fix the FAILs above before running the dilution.")
            _save(checks, p.output_dir)
            return 1

        res = run_dilution(p, store=store)
        paths = write_outputs(res, p.output_dir)
        log.info("\n%s", format_report(res))

        out = check_outputs(store, p, res)
        log.info("Output checks\n\n%s", format_checks(out))
        checks += out
        _save(checks, p.output_dir)
    finally:

        store.close()
    log.info("Outputs:\n  %s", "\n  ".join(paths.values()))
    failed = sum(c.status == "FAIL" for c in checks)
    summary = (f"{'FAILED' if failed else 'PASSED'}: {failed} failed checks, "
               f"{sum(c.status == 'WARN' for c in checks)} warnings")
    (log.error if failed else log.info)(summary)
    return 1 if failed else 0

def main(argv) -> int:
    setup_logging()
    try:
        return run_checks(load_params(argv))
    except FileNotFoundError as e:
        log.error("%s", e)
    except PermissionError as e:
        log.error("Cannot read or write %s: %s", e.filename, e.strerror)
    except (ValueError, KeyError) as e:
        log.error("%s", e)
    except MemoryError:
        log.error("Ran out of memory. Use fewer threads or free some RAM.")
    except Exception:
        log.exception("The model check failed")
    return 1

def _save(checks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(os.path.join(out_dir, "model_checks.txt"), "w") as f:
            f.write(format_checks(checks))
        pd.DataFrame([c.__dict__ for c in checks]).to_csv(
            os.path.join(out_dir, "model_checks.csv"), index=False)
    except OSError as e:
        log.error("Could not save the check results to %s: %s", out_dir, e)
        raise

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
