from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .checks import check_inputs, check_outputs, format_checks
from .core import (ELOS_AXES, ORE_RULES, OUTSIDE, DilutionCancelled, DilutionParams,
                   block_outline, def_col, format_report, read_header,
                   run_dilution, worker_pool, write_outputs)
from .extras import (HISTORY_FILE, SENS_MODEL_DIR, SENS_PARAMS, append_history,
                     format_sensitivity, parse_list, read_history,
                     run_sensitivity, value_range, write_sensitivity)
from .merge import GROUP_ON
from .merge import RULES as MERGE_RULES
from .previous import (LoadedRun, output_folder, read_settings,
                       rebuild_sensitivity)
from .preview import (arrow, plan_grid, plan_layers, region_colour, region_table, write_pdf)
from .regions import as_regions, load_regions, order_regions
from .resources import detect, recommend_threads
from .store import BlockStore
from . import theme as T

NONE = "(none)"

COLUMN_FIELDS = [
    ("x_col", "X centroid", ["XC", "X", "XCENTRE", "EAST"], False),
    ("y_col", "Y centroid", ["YC", "Y", "YCENTRE", "NORTH"], False),
    ("z_col", "Z centroid", ["ZC", "Z", "ZCENTRE", "RL", "ELEV"], False),
    ("grade_col", "Grade (AU)", ["AU", "AU_PPM", "AU_GPT", "GOLD"], False),
    ("rescat_col", "Resource category", ["RESCAT", "RES_CAT", "CLASS"], False),
    ("density_col", "Density", ["DENSITY", "SG", "BD", "DENS"], True),
    ("oxide_col", "Material type (OXIDE)", ["OXIDE", "OX", "WEATH", "MATTYPE"], True),
    ("xinc_col", "X block size (sub-blocked)",
     ["XINC", "_XINC", "XSIZE", "X_SIZE", "XLENGTH", "XDIM", "DX"], True),
    ("yinc_col", "Y block size (sub-blocked)",
     ["YINC", "_YINC", "YSIZE", "Y_SIZE", "YLENGTH", "YDIM", "DY"], True),
    ("zinc_col", "Z block size (sub-blocked)",
     ["ZINC", "_ZINC", "ZSIZE", "Z_SIZE", "ZLENGTH", "ZDIM", "DZ"], True),
]

NUM_FIELDS = [
    ("cutoff", "Cut-off grade (g/t)", "0.5"),
    ("smu_x", "SMU X (m)", "10"),
    ("smu_y", "SMU Y (m, blank = parent)", ""),
    ("smu_z", "SMU Z (m, blank = parent)", ""),
    ("step_x", "X step (m)", "1"),
    ("step_y", "Y step (m)", "1"),
    ("elos", "Edge dilution per side (m)", "0"),
    ("block_x", "Parent X (m, blank = infer)", ""),
    ("block_y", "Parent Y (m, blank = infer)", ""),
    ("block_z", "Parent Z (m, blank = infer)", ""),
    ("origin_x", "Origin X (m, blank = infer)", ""),
    ("origin_y", "Origin Y (m, blank = infer)", ""),
    ("origin_z", "Origin Z (m, blank = infer)", ""),
    ("default_density", "Default density (t/m\u00b3)", "2.7"),
    ("skip_rows", "Header rows to skip (CSV)", "0"),
]

MODEL_TYPES = [("Block models", "*.csv *.dm *.dmx"), ("CSV files", "*.csv"),
               ("Datamine tables", "*.dm *.dmx"), ("All files", "*.*")]
REGION_TYPES = [("Regions", "*.csv *.str"), ("CSV files", "*.csv"),
                ("Surpac strings", "*.str"), ("All files", "*.*")]

_log = logging.getLogger(__name__)

COLUMN_TREATMENT = {
    "x_col": "position", "y_col": "position", "z_col": "position",
    "grade_col": "average by tonnes", "rescat_col": "dominant by metal",
    "density_col": "tonnes \u00f7 volume", "oxide_col": "dominant by mass",
    "xinc_col": "cell size", "yinc_col": "cell size", "zinc_col": "cell size",
}
REQUIRED_NUMBERS = ("cutoff", "smu_x", "step_x", "step_y")
WHOLE_NUMBERS = ("skip_rows",)

class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):

        self.q.put(f"{record.levelname}: {record.getMessage()}")

class DilutionApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        T.apply(master)
        super().__init__(master, padding=8, style="Base.TFrame")
        master.title("Open pit SMU dilution")
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.q: queue.Queue = queue.Queue()
        self.columns: list[str] = []
        self.col_vars: dict[str, tk.StringVar] = {}
        self.col_boxes: dict[str, ttk.Combobox] = {}
        self.num_vars: dict[str, tk.StringVar] = {}
        self.cancel_event: threading.Event | None = None
        self.regions: list = []
        self.last_result = None
        self.previews: list = []
        self.t0 = time.time()
        master.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build()
        h = _QueueHandler(self.q)
        h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logging.getLogger("dilution").addHandler(h)
        logging.getLogger("dilution").setLevel(logging.INFO)
        self.after(100, self._drain)

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.tabs = ttk.Notebook(self)
        self.tabs.grid(row=0, column=0, sticky="ew")
        main = ttk.Frame(self.tabs, padding=8)
        regs = ttk.Frame(self.tabs, padding=8)
        opts = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(main, text="Model and parameters")
        self.tabs.add(regs, text="Regions")
        self.tabs.add(opts, text="Options")
        self._build_main(main)
        self._build_regions(regs)
        self._build_options(opts)

        banner = ttk.Frame(self, style="Banner.TFrame", padding=(8, 6))
        banner.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        banner.columnconfigure(1, weight=1)
        ttk.Label(banner, text="Run mode", style="BannerKey.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12))
        self.mode_text = tk.StringVar()
        ttk.Label(banner, textvariable=self.mode_text, style="BannerMode.TLabel").grid(
            row=0, column=1, sticky="w")
        for v in (self.mode_var, self.elos_axes_var, self.rule_var, self.num_vars["elos"]):
            v.trace_add("write", lambda *a: self._show_mode())
        self._show_mode()

        btns = ttk.Frame(self, style="Base.TFrame")
        btns.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(btns, text="Load settings...", command=self._load_settings).pack(side="left")
        ttk.Button(btns, text="Save settings...",
                   command=self._save_settings).pack(side="left", padx=4)
        ttk.Button(btns, text="Plan preview...",
                   command=self._preview).pack(side="left", padx=(12, 0))
        ttk.Button(btns, text="Load previous run...",
                   command=self._load_previous).pack(side="left", padx=4)
        ttk.Button(btns, text="Run history...", command=self._history).pack(side="left", padx=4)
        self.run_btn = ttk.Button(btns, text="Run dilution (F5)", command=self._run,
                                  style="Accent.TButton")
        self.run_btn.pack(side="right")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self._cancel)
        self.cancel_btn.pack(side="right", padx=4)
        self.cancel_btn.state(["disabled"])
        ttk.Button(btns, text="Open output folder",
                   command=self._open_output).pack(side="right", padx=4)

        prog = ttk.Frame(self, style="Base.TFrame")
        prog.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        prog.columnconfigure(0, weight=1)
        self.bar = ttk.Progressbar(prog, mode="determinate")
        self.bar.grid(row=0, column=0, sticky="ew")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(prog, textvariable=self.status_var, width=36, style="Base.TLabel").grid(
            row=0, column=1, sticky="e", padx=(8, 0))

        self.log = tk.Text(self, height=8, width=90, font=T.fonts.mono, background=T.BASE,
                           foreground=T.TEXT, highlightbackground=T.BORDER,
                           padx=6, pady=4, wrap="word")
        for key, st in T.STATUS.items():
            self.log.tag_configure(key, foreground=st["fg"], background=st["bg"] or "",
                                   font=T.fonts.bold)
        self.log.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        self.rowconfigure(4, weight=1)
        master = self.winfo_toplevel()
        master.bind("<F5>", lambda e: self.run_btn.instate(["!disabled"]) and self._run())
        master.bind("<Escape>",
                    lambda e: self.cancel_btn.instate(["!disabled"]) and self._cancel())

    def _show_mode(self):
        rule = ORE_RULES.get(self.rule_var.get(), self.rule_var.get())
        axes = ELOS_AXES.get(self.elos_axes_var.get(), self.elos_axes_var.get())
        elos = self.num_vars["elos"].get().strip() or "0"
        text = (f"{self.mode_var.get()}   \u2022   edge dilution {elos} m in {axes}"
                f"   \u2022   {rule}")
        self.mode_text.set(text)
        self.winfo_toplevel().title(f"Open pit SMU dilution: {self.mode_var.get()}, "
                                    f"edge dilution in {axes}")

    def _build_main(self, f):
        f.columnconfigure(0, weight=1)
        files = ttk.LabelFrame(f, text="Files", padding=8)
        files.grid(row=0, column=0, sticky="ew")
        files.columnconfigure(1, weight=1)
        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()
        ttk.Label(files, text="Block model (CSV or .dm)").grid(row=0, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.in_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(files, text="Browse...", command=self._pick_input).grid(row=0, column=2)
        ttk.Label(files, text="Output folder").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(files, textvariable=self.out_var).grid(row=1, column=1, sticky="ew",
                                                         padx=8, pady=(4, 0))
        ttk.Button(files, text="Browse...", command=self._pick_output).grid(
            row=1, column=2, pady=(4, 0))

        mode = ttk.LabelFrame(f, text="Run mode", padding=8)
        mode.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.mode_var = tk.StringVar(value="MII")
        self.rule_var = tk.StringVar(value="recover")
        self.elos_axes_var = tk.StringVar(value="xy")
        rows = (("Dilution run", self.mode_var,
                 (("MII", "MII (RESCAT = 4 at zero grade)"),
                  ("MI", "MI (RESCAT \u2265 3 at zero grade)"))),
                ("Edge dilution in", self.elos_axes_var, tuple(ELOS_AXES.items())),
                ("Ore rule", self.rule_var, tuple(ORE_RULES.items())))
        for r, (label, var, choices) in enumerate(rows):
            ttk.Label(mode, text=label, font=T.fonts.bold).grid(
                row=r, column=0, sticky="w", padx=(0, 12), pady=1)
            for c, (key, text) in enumerate(choices, 1):
                ttk.Radiobutton(mode, text=text, value=key, variable=var).grid(
                    row=r, column=c, sticky="w", padx=(0, 16), pady=1)

        mid = ttk.Frame(f)
        mid.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)

        cols = ttk.LabelFrame(mid, text="Block model columns", padding=8)
        cols.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        cols.columnconfigure(1, weight=1)
        ttk.Label(cols, text="Field", style="Help.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(cols, text="Column", style="Help.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(cols, text="Treated as", style="Help.TLabel").grid(
            row=0, column=2, sticky="w", padx=(8, 0))
        for r, (key, label, _, optional) in enumerate(COLUMN_FIELDS, 1):
            ttk.Label(cols, text=label.replace(" (sub-blocked)", "")).grid(
                row=r, column=0, sticky="w")
            v = tk.StringVar(value=NONE if optional else "")
            cb = ttk.Combobox(cols, textvariable=v, state="readonly", width=12)
            cb.grid(row=r, column=1, sticky="ew", pady=1)
            ttk.Label(cols, text=COLUMN_TREATMENT[key], style="Help.TLabel").grid(
                row=r, column=2, sticky="w", padx=(8, 0))
            self.col_vars[key] = v
            self.col_boxes[key] = cb
        ttk.Label(cols, text="Block sizes are for sub-blocked models. Extra "
                  "attributes are set on the Options tab.", style="Help.TLabel",
                  wraplength=330).grid(row=len(COLUMN_FIELDS) + 1, column=0,
                                       columnspan=3, sticky="w", pady=(6, 0))

        par = ttk.LabelFrame(mid, text="Dilution parameters", padding=8)
        par.grid(row=0, column=1, sticky="nsew")
        vcmd = (self.register(self._check_number), "%W")
        defaults = {key: default for key, _, default in NUM_FIELDS}
        self.num_entries: dict[str, ttk.Entry] = {}

        def entry(key, row, col):
            v = tk.StringVar(value=defaults[key])
            e = ttk.Entry(par, textvariable=v, width=7, justify="right",
                          validate="focusout", validatecommand=vcmd)
            e.grid(row=row, column=col, sticky="ew", pady=2, padx=(6, 0))
            self.num_vars[key] = v
            self.num_entries[key] = e

        for c, axis in enumerate("XYZ", 1):
            ttk.Label(par, text=axis, style="Help.TLabel", anchor="e").grid(
                row=0, column=c, sticky="e")
        grid_rows = (("SMU (m) *", ("smu_x", "smu_y", "smu_z")),
                     ("Step (m) *", ("step_x", "step_y", None)),
                     ("Parent block (m)", ("block_x", "block_y", "block_z")),
                     ("Model origin (m)", ("origin_x", "origin_y", "origin_z")))
        for r, (label, keys) in enumerate(grid_rows, 1):
            ttk.Label(par, text=label).grid(row=r, column=0, sticky="w")
            for c, key in enumerate(keys, 1):
                if key:
                    entry(key, r, c)
        r = len(grid_rows) + 1
        ttk.Label(par, text="Blank SMU Y or Z = parent cell; blank parent or origin = "
                  "inferred from the model.", style="Help.TLabel", wraplength=330).grid(
            row=r, column=0, columnspan=4, sticky="w", pady=(0, 6))
        singles = (("cutoff", "Cut-off grade (g/t) *"), ("elos", "Edge dilution per side (m)"),
                   ("default_density", "Default density (t/m\u00b3)"),
                   ("skip_rows", "Header rows to skip (CSV)"))
        for r, (key, label) in enumerate(singles, r + 1):
            ttk.Label(par, text=label).grid(row=r, column=0, columnspan=3, sticky="w")
            entry(key, r, 3)
        r += 1
        self.invalid_var = tk.StringVar()
        self.invalid_lbl = ttk.Label(par, textvariable=self.invalid_var, style="Fail.TLabel",
                                     wraplength=330)
        self.invalid_lbl.grid(row=r, column=0, columnspan=4, sticky="w", pady=(6, 0))
        self.invalid_lbl.grid_remove()

        ttk.Label(f, text="SMU and parent cell must share a sub-cell size (e.g. 7.5 m "
                  "in 10 m: 2.5 m). SMU X must be a whole number of X steps and SMU Y "
                  "of Y steps. Ore blocks grow across strips in Y steps to take partial "
                  "ore (a Y step equal to SMU Y turns this off). Edge dilution per side "
                  "is a whole number of X steps, and of Y steps in X and Y; it widens "
                  "the ends of each ore zone in X and block edges facing waste in Y. "
                  "Waste between ore keeps a whole SMU.",
                  style="Help.TLabel", wraplength=880).grid(row=3, column=0, sticky="w",
                                                            pady=(8, 0))

    def _check_number(self, widget: str) -> bool:
        problems = []
        for key, label, _ in NUM_FIELDS:
            text = self.num_vars[key].get().strip()
            name = label.split(" (")[0]
            if not text:
                if key in REQUIRED_NUMBERS:
                    problems.append(f"{name} is required")
                continue
            try:
                (int if key in WHOLE_NUMBERS else float)(text)
            except ValueError:
                kind = "a whole number" if key in WHOLE_NUMBERS else "a number"
                problems.append(f"{name}: enter {kind}")
        if problems:
            icon = T.STATUS["FAIL"]["icon"]
            self.invalid_var.set("\n".join(f" {icon} {p} " for p in problems))
            self.invalid_lbl.grid()
        else:
            self.invalid_var.set("")
            self.invalid_lbl.grid_remove()
        return True

    def _build_options(self, f):
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

        self.res = detect()
        perf = ttk.LabelFrame(f, text="Performance", padding=8)
        perf.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(perf, text="Threads").grid(row=0, column=0, sticky="w")
        rec = recommend_threads(self.res)
        tv = tk.StringVar(value=str(rec))
        ttk.Spinbox(perf, from_=1, to=max(512, self.res.threads), textvariable=tv,
                    width=5).grid(row=0, column=1, sticky="w", padx=4)
        self.num_vars["threads"] = tv
        self.res_var = tk.StringVar()
        ttk.Label(perf, textvariable=self.res_var, style="Help.TLabel").grid(
            row=0, column=2, sticky="w", padx=8)
        self._show_resources(rec)

        ex = ttk.LabelFrame(f, text="Extra attributes to carry", padding=8)
        ex.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        for c in range(3):
            ex.columnconfigure(c, weight=1)
        self.extra_lists = {}
        for c, (key, label) in enumerate(
                (("extra_numeric", "Average by tonnes\n(grades, e.g. AG, CU, S)"),
                 ("extra_categorical", "Dominant by mass\n(codes, e.g. LITH, DOMAIN)"),
                 ("extra_metal", "Dominant by metal (ounces)\n(codes, e.g. STOCKPILE)"))):
            fr = ttk.Frame(ex)
            fr.grid(row=0, column=c, sticky="ew", padx=(0, 8))
            fr.columnconfigure(0, weight=1)
            ttk.Label(fr, text=label).grid(row=0, column=0, sticky="w")
            lb = tk.Listbox(fr, selectmode="extended", exportselection=False,
                            height=4, font=T.fonts.body)
            ttk.Button(fr, text="Clear", style="Small.TButton", width=6,
                       command=lambda lb=lb: lb.selection_clear(0, "end")).grid(
                row=0, column=1, sticky="se", pady=(0, 2))
            lb.grid(row=1, column=0, columnspan=2, sticky="ew")
            sb = ttk.Scrollbar(fr, orient="vertical", command=lb.yview)
            sb.grid(row=1, column=2, sticky="ns")
            lb["yscrollcommand"] = sb.set
            self.extra_lists[key] = lb
        ttk.Label(ex, text="Ctrl-click or Shift-click to pick several. RESCAT is always carried by dominant metal: waste is "
                  "RESCAT 4, so an ore block takes the category holding most of "
                  "its ounces, not most of its tonnes. OXIDE is carried by mass. "
                  "Where a block has no metal, codes fall back to dominant by mass.",
                  style="Help.TLabel", wraplength=880).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        pd = ttk.LabelFrame(f, text="Partial dilution by attribute", padding=8)
        pd.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        pd.columnconfigure(1, weight=1)
        ttk.Label(pd, text="Attribute column").grid(row=0, column=0, sticky="w")
        self.dilute_attr_col_var = tk.StringVar(value=NONE)
        self.dilute_attr_col_box = ttk.Combobox(
            pd, textvariable=self.dilute_attr_col_var, state="readonly", width=14)
        self.dilute_attr_col_box["values"] = [NONE]
        self.dilute_attr_col_box.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(pd, text="Values to dilute (comma-separated)").grid(
            row=1, column=0, sticky="w")
        self.dilute_attr_values_var = tk.StringVar()
        ttk.Entry(pd, textvariable=self.dilute_attr_values_var, width=40).grid(
            row=1, column=1, sticky="ew", padx=4)
        ttk.Label(pd, text="Only blocks whose dominant attribute (by mass) is "
                  "one of these values are diluted. Unselected blocks keep their "
                  "undiluted grade. Leave the column as (none) to dilute everything.",
                  style="Help.TLabel", wraplength=880).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        sens = ttk.LabelFrame(f, text="Sensitivity", padding=8)
        sens.grid(row=3, column=0, columnspan=2, sticky="ew")
        for c, h in ((0, "Run"), (1, "From"), (2, "To"), (3, "Step"),
                     (5, "Values to run")):
            ttk.Label(sens, text=h, style="Help.TLabel").grid(row=0, column=c, sticky="w")
        self.sens = {}
        defaults = {"cutoff": ("0.3", "0.8", "0.1")}
        for r, (key, label) in enumerate(SENS_PARAMS.items(), 1):
            on = tk.BooleanVar(value=False)
            ttk.Checkbutton(sens, text=label, variable=on).grid(row=r, column=0, sticky="w")
            v = {"on": on}
            for c, part in enumerate(("from", "to", "step"), 1):
                v[part] = tk.StringVar(value=defaults.get(key, ("", "", ""))[c - 1])
                ttk.Entry(sens, textvariable=v[part], width=7).grid(
                    row=r, column=c, sticky="w", padx=(0, 4), pady=1)
            ttk.Button(sens, text="Add range \u203a", style="Small.TButton",
                       command=lambda k=key: self._fill_range(k)).grid(
                row=r, column=4, sticky="w", padx=(0, 6))
            txt = tk.Text(sens, height=1, width=30, font=T.fonts.body)
            txt.grid(row=r, column=5, sticky="ew", pady=1)
            txt.bind("<<Paste>>", self._paste_values)
            v["list"] = txt
            self.sens[key] = v
        sens.columnconfigure(5, weight=1)
        ttk.Label(sens, text="Runs after the main run and writes one "
                  "sensitivity_<parameter>_<run>.csv per parameter, each run's diluted "
                  "model in sensitivity_models (.csv and Datamine .dm, or .dm only) and "
                  "sensitivity_benches_<run>.csv: tonnes, grade and ounces by bench, "
                  "undiluted and for each run. Only the "
                  "values in "
                  "the list are run: type them, paste an Excel row or column, or use "
                  "From/To/Step and Add range, then edit the list if needed. "
                  "Each value is a separate run of the main settings "
                  "with only that parameter changed. An SMU X that isn't a whole "
                  "number of X steps uses the main X step split evenly until it fits "
                  "(shown as STEP_X), and SMU Y likewise with the Y step (STEP_Y). "
                  "An X step must divide the main SMU X and a Y step the main SMU Y. "
                  "SMU sizes must share a sub-cell size with the parent cell.",
                  style="Help.TLabel", wraplength=880).grid(
            row=len(SENS_PARAMS) + 1, column=0, columnspan=6, sticky="w", pady=(4, 0))

        mg = ttk.LabelFrame(f, text="Merge diluted sub-cells", padding=8)
        mg.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        mg.columnconfigure(0, weight=1)
        self.merge_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mg, text="Merge sub-cells of the diluted model that lie in "
                        "one parent cell and group together",
                        variable=self.merge_var).grid(row=0, column=0, sticky="w")
        ttk.Label(mg, text="Cells merge only inside a parent cell, so parent cell "
                  "sizes and boundaries are kept, and only where the merged cell "
                  "is a box. Diluted attributes (D_ fields, ore or waste, block) "
                  "always group: they must be identical. For each initial "
                  "attribute, choose Group on (must be identical) or how to "
                  "calculate the merged value. Tonnes, density, metal and other "
                  "supporting fields are calculated after the cells are merged. "
                  "The combine rules need the trimmed diluted model: with working "
                  "columns kept, only identical cells merge.",
                  style="Help.TLabel", wraplength=880).grid(
            row=1, column=0, sticky="w", pady=(2, 4))
        self.merge_table = ttk.Frame(mg)
        self.merge_table.grid(row=2, column=0, sticky="w")
        self.merge_choice = {}
        self._refresh_merge_attrs()
        for key in ("grade_col", "rescat_col", "oxide_col"):
            self.col_vars[key].trace_add("write", lambda *_: self._refresh_merge_attrs())
        for lb in self.extra_lists.values():
            lb.bind("<<ListboxSelect>>", lambda _e: self._refresh_merge_attrs(), add="+")

        aud = ttk.LabelFrame(f, text="Checks and audit", padding=8)
        aud.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.check_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(aud, text="Check the model and results (writes "
                        "model_checks.txt)", variable=self.check_var).grid(
            row=0, column=0, sticky="w")
        self.hist_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(aud, text="Log each run to the run history",
                        variable=self.hist_var).grid(row=1, column=0, sticky="w")
        ttk.Label(aud, text=f"History is kept in {HISTORY_FILE} and in "
                  "run_history.csv in the output folder", style="Help.TLabel",
                  wraplength=900).grid(row=2, column=0, sticky="w")
        self.audit_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(aud, text="Keep working columns in the diluted model "
                        "(tonnes, metal, ids, tonnes by RESCAT; larger files)",
                        variable=self.audit_var).grid(row=3, column=0, sticky="w")
        self.dm_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(aud, text="Write diluted models as Datamine .dm only "
                        "(no CSV copy; saves disk)",
                        variable=self.dm_only_var).grid(row=4, column=0, sticky="w")

    def _build_regions(self, f):
        f.columnconfigure(0, weight=1)
        top = ttk.Frame(f)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="Load regions...", command=self._load_regions).pack(side="left")
        ttk.Button(top, text="Remove", command=self._remove_region).pack(side="left", padx=4)
        ttk.Button(top, text="Clear", command=self._clear_regions).pack(side="left")
        ttk.Label(top, text="Priority of selected:").pack(side="left", padx=(16, 2))
        self.prio_var = tk.StringVar(value="1")
        ttk.Spinbox(top, from_=1, to=999, textvariable=self.prio_var,
                    width=5).pack(side="left")
        ttk.Button(top, text="Set", command=self._set_priority).pack(side="left", padx=4)
        ttk.Button(top, text="Preview regions...",
                   command=self._preview_regions).pack(side="right")

        cols = ("REGION", "PRIORITY", "AZIMUTH", "LENGTH", "WIDTH")
        self.reg_tree = ttk.Treeview(f, columns=cols, show="headings", height=7)
        for c, w in zip(cols, (160, 80, 90, 100, 100)):
            self.reg_tree.heading(c, text=_heading(c))
            self.reg_tree.column(c, width=w, anchor="w" if c == "REGION" else "e")
        self.reg_tree.grid(row=1, column=0, sticky="ew", pady=8)

        opt = ttk.Frame(f)
        opt.grid(row=2, column=0, sticky="ew")
        ttk.Label(opt, text="Material outside every region:").pack(side="left")
        self.outside_var = tk.StringVar(value="model")
        for key, text in OUTSIDE.items():
            ttk.Radiobutton(opt, text=text, value=key,
                            variable=self.outside_var).pack(side="left", padx=4)
        ttk.Label(opt, text="Piece size (m, blank = auto)").pack(side="left", padx=(16, 2))
        self.sample_var = tk.StringVar(value="")
        ttk.Entry(opt, textvariable=self.sample_var, width=7).pack(side="left")
        ttk.Label(f, text="Each region is a rectangle of 4 corners: corner 1 to 2 "
                  "is the dilution direction (across strike), 2 to 3 runs along "
                  "strike. Load a CSV with REGION, PRIORITY, X1, Y1 ... X4, Y4 (or "
                  "one row per point: REGION, X, Y), or a Surpac .str with one "
                  "string per region. Where regions overlap, priority 1 dilutes "
                  "the overlap. Region edges act as model edges for the waste "
                  "rule. Preview before running to check the regions cover the ore.",
                  style="Help.TLabel", wraplength=880).grid(row=3, column=0,
                                                          sticky="w", pady=(6, 0))

    def _refresh_regions(self):
        t = self.reg_tree
        t.delete(*t.get_children())
        for r in self.regions:
            t.insert("", "end", iid=r.name, values=(
                r.name, r.priority, f"{r.azimuth:05.1f}", f"{r.length:,.1f}",
                f"{r.width:,.1f}"))

    def _load_regions(self):
        path = filedialog.askopenfilename(filetypes=REGION_TYPES)
        if not path:
            return
        try:
            regions = load_regions(path)
        except (OSError, ValueError, KeyError) as e:
            _log.error("Could not load regions from %s: %s", path, e)
            messagebox.showerror("Regions", f"Could not load the regions:\n{e}")
            return
        self.regions = regions
        self._refresh_regions()
        self._write(f"Loaded {len(regions)} region(s) from {path}")

    def _remove_region(self):
        gone = set(self.reg_tree.selection())
        self.regions = [r for r in self.regions if r.name not in gone]
        self._refresh_regions()

    def _clear_regions(self):
        self.regions = []
        self._refresh_regions()

    def _set_priority(self):
        try:
            prio = int(self.prio_var.get())
            if prio < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Regions", "Priority must be a whole number of 1 or more")
            return
        sel = set(self.reg_tree.selection())
        self.regions = [dataclasses.replace(r, priority=prio) if r.name in sel else r
                        for r in self.regions]
        self._refresh_regions()

    def _preview_regions(self):
        try:
            p = self._params()
            if p.cutoff is None:
                raise ValueError("Enter a cut-off grade")
            regions = order_regions(as_regions(p.regions))
        except (ValueError, KeyError) as e:
            messagebox.showerror("Region preview", str(e))
            return
        self.status_var.set("Reading the model for the region preview...")

        def work():
            try:
                plan = plan_grid(None, p, progress=lambda s: self.q.put(("status", s)))
                self.q.put(("preview", plan, regions, p))
                self.q.put(("status", "Region preview ready"))
            except Exception as e:
                _log.exception("Region preview failed")
                self.q.put(f"ERROR: region preview: {e}")
                self.q.put(("status", "Region preview failed, see the log"))

        threading.Thread(target=work, daemon=True).start()

    def _show_resources(self, rec: int, note: str = ""):
        self.res_var.set(f"Detected {self.res.describe()}. "
                         f"Recommended: {rec}{note}")

    def _pick_input(self):
        path = filedialog.askopenfilename(filetypes=MODEL_TYPES)
        if path:
            self.in_var.set(path)
            if not self.out_var.get():
                self.out_var.set(os.path.join(os.path.dirname(path), "dilution_output"))
            self._load_columns()

    def _pick_output(self):
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def _load_columns(self) -> bool:
        try:
            skip = int(self.num_vars["skip_rows"].get() or 0)
            self.columns = read_header(self.in_var.get(), skip)
        except Exception as e:
            messagebox.showerror("Block model", f"Could not read the block model:\n{e}")
            return False
        upper = {c.upper(): c for c in self.columns}
        for key, _, guesses, optional in COLUMN_FIELDS:
            cb = self.col_boxes[key]
            cb["values"] = ([NONE] if optional else []) + self.columns
            guess = next((upper[g] for g in guesses if g in upper), None)
            self.col_vars[key].set(guess or (NONE if optional else ""))
        for lb in self.extra_lists.values():
            lb.delete(0, "end")
            for c in self.columns:
                lb.insert("end", c)
        self.dilute_attr_col_box["values"] = [NONE] + self.columns
        self._write(f"Columns: {', '.join(self.columns)}")
        try:
            size = os.path.getsize(self.in_var.get())
            self.res = detect()
            rec = recommend_threads(self.res, size)
            self.num_vars["threads"].set(str(rec))
            self._show_resources(rec, f" for this {size / 1024 ** 2:,.0f} MB model")
        except OSError as e:
            _log.debug("Could not size the model for a thread count: %s", e)
        return True

    def _merge_attributes(self) -> list[tuple[str, str]]:
        out = []
        for key, kind in (("grade_col", "num"), ("rescat_col", "cat"),
                          ("oxide_col", "cat")):
            name = self.col_vars[key].get().strip()
            if name and name != NONE:
                out.append((name, kind))
        for key, kind in (("extra_numeric", "num"), ("extra_categorical", "cat"),
                          ("extra_metal", "cat")):
            out += [(n, kind) for n in self._extras(key)]
        seen, uniq = set(), []
        for name, kind in out:
            if name.upper() not in seen:
                seen.add(name.upper())
                uniq.append((name, kind))
        return uniq

    def _refresh_merge_attrs(self):
        for w in self.merge_table.winfo_children():
            w.destroy()
        for c, h in enumerate(("Initial attribute", "Group on, or combine by")):
            ttk.Label(self.merge_table, text=h, style="Help.TLabel").grid(
                row=0, column=c, sticky="w", padx=(0, 12))
        for r, (name, kind) in enumerate(self._merge_attributes(), 1):
            var = self.merge_choice.setdefault(name, tk.StringVar(value=GROUP_ON))
            rules = [v for k, v in MERGE_RULES.items() if kind == "num" or k.startswith("dom")]
            if var.get() not in [GROUP_ON, *rules]:
                var.set(GROUP_ON)
            ttk.Label(self.merge_table, text=name).grid(
                row=r, column=0, sticky="w", padx=(0, 12), pady=1)
            ttk.Combobox(self.merge_table, textvariable=var, state="readonly",
                         width=22, values=[GROUP_ON, *rules]).grid(
                row=r, column=1, sticky="w", pady=1)

    def _merge_rules(self) -> dict:
        by_label = {v: k for k, v in MERGE_RULES.items()}
        out = {}
        for name, _ in self._merge_attributes():
            rule = by_label.get(self.merge_choice[name].get()) if name in self.merge_choice else None
            if rule:
                out[name] = rule
        return out

    def _extras(self, key) -> list[str]:
        lb = self.extra_lists[key]
        return [lb.get(i) for i in lb.curselection()]

    def _set_extras(self, key, names):
        lb = self.extra_lists[key]
        lb.selection_clear(0, "end")
        items = lb.get(0, "end")
        for n in names:
            if n in items:
                lb.selection_set(items.index(n))

    def _params(self) -> DilutionParams:
        def num(key, cast=float):
            s = self.num_vars[key].get().strip()
            return cast(s) if s else None

        cols = {k: (v.get() if v.get() not in ("", NONE) else None)
                for k, v in self.col_vars.items()}
        for key, label, _, optional in COLUMN_FIELDS:
            if not optional and not cols[key]:
                raise ValueError(f"Select a column for {label}")
        if not self.in_var.get():
            raise ValueError("Select a block model file")
        if not os.path.isfile(self.in_var.get()):
            raise ValueError(f"Block model not found:\n{self.in_var.get()}")
        if not self.out_var.get():
            raise ValueError("Select an output folder")
        return DilutionParams(
            input_path=self.in_var.get(), output_dir=self.out_var.get(),
            **cols,
            default_density=num("default_density") or 2.7,
            block_x=num("block_x"), block_y=num("block_y"), block_z=num("block_z"),
            origin_x=num("origin_x"), origin_y=num("origin_y"),
            origin_z=num("origin_z"),
            cutoff=num("cutoff"), smu_x=num("smu_x"),
            smu_y=num("smu_y"), smu_z=num("smu_z"),
            step_x=num("step_x"), step_y=num("step_y"), mode=self.mode_var.get(),
            skip_rows=num("skip_rows", int) or 0,
            threads=num("threads", int) or 1,
            extra_numeric=self._extras("extra_numeric"),
            extra_categorical=self._extras("extra_categorical"),
            extra_metal=self._extras("extra_metal"),
            ore_rule=self.rule_var.get(),
            dilute_attr_col=(self.dilute_attr_col_var.get()
                             if self.dilute_attr_col_var.get() not in ("", NONE)
                             else None),
            dilute_attr_values=[v.strip() for v in
                                self.dilute_attr_values_var.get().split(",")
                                if v.strip()],
            elos_axes=self.elos_axes_var.get(),
            elos=num("elos") or 0.0,
            regions=list(self.regions),
            outside=self.outside_var.get(),
            sample_size=float(self.sample_var.get()) if self.sample_var.get().strip() else None,
            audit_fields=self.audit_var.get(),
            merge_subcells=self.merge_var.get(),
            merge_rules=self._merge_rules(),
        )

    def _run(self):
        try:
            p = self._params()
            if (p.cutoff is None or p.smu_x is None or p.step_x is None
                    or p.step_y is None):
                raise ValueError("Cut-off, SMU X, X step size and Y step size "
                                 "are required")
            p.validate()
            sens = self._sens_values()
        except Exception as e:
            messagebox.showerror("Parameters", str(e))
            return
        self.run_btn.state(["disabled"])
        self.cancel_btn.state(["!disabled"])
        self._write("-" * 60)
        self.bar["value"] = 0
        self.t0 = time.time()
        self.status_var.set("Loading block model...")
        do_checks, do_hist = self.check_var.get(), self.hist_var.get()
        dm_only = self.dm_only_var.get()
        cancel = self.cancel_event = threading.Event()

        self._release_result()

        def work():
            pool = store = res = None
            try:

                store = BlockStore.build(p, progress=lambda s: self.q.put(("status", s)))
                checks = []
                if do_checks:
                    self.q.put(("status", "Checking the model..."))
                    checks = check_inputs(store, p)
                    self.q.put("Input model checks\n" + format_checks(checks))
                    if any(c.status == "FAIL" for c in checks):
                        self.q.put("Fix the FAILs above before running the dilution.")
                        self.q.put(("status", "Model checks failed, see the log"))
                        return
                if cancel.is_set():
                    raise DilutionCancelled("Run cancelled")
                pool = worker_pool(p.threads)
                res = run_dilution(
                    p, progress=lambda s: self.q.put(s), store=store,
                    bench_progress=lambda d, n: self.q.put(("progress", d, n)),
                    cancel=cancel, pool=pool)
                self.q.put(("result", res))
                self.q.put(("status", "Writing outputs..."))
                paths = write_outputs(res, p.output_dir, diluted_csv=not dm_only)
                if do_hist:
                    append_history(res, time.time() - self.t0)
                self.q.put(format_report(res))
                if do_checks:
                    out = check_outputs(store, p, res)
                    self.q.put("Output checks\n" + format_checks(out))
                    with open(os.path.join(p.output_dir, "model_checks.txt"), "w") as f:
                        f.write(format_checks(checks + out))
                if sens:
                    self.q.put(("status", "Running sensitivity..."))
                    tables = run_sensitivity(
                        p, store, sens, progress=lambda s: self.q.put(s),
                        step=lambda d, n: self.q.put(("sens", d, n)), cancel=cancel,
                        base=res, pool=pool,
                        model_dir=os.path.join(p.output_dir, SENS_MODEL_DIR),
                        model_csv=not dm_only)
                    paths.update(write_sensitivity(tables, p.output_dir, p.mode,
                                                   res.params))
                    self.q.put(format_sensitivity(tables))

                if res.spool is not None:
                    res.spool.drop_model()
                self.q.put("Outputs:\n  " + "\n  ".join(paths.values()))
                self.q.put(("status", f"Done in {time.time() - self.t0:,.0f} s"))
            except DilutionCancelled:
                self.q.put("Run cancelled.")
                self.q.put(("status", "Cancelled"))
            except Exception as e:
                _log.exception("The dilution run failed")
                self.q.put(f"ERROR: {e}")
                self.q.put(("status", "Failed, see the log"))
            finally:
                if pool is not None:
                    pool.close(wait=False)
                if res is None and store is not None:
                    store.close()
                self.q.put(None)

        threading.Thread(target=work, daemon=True).start()

    def _paste_values(self, event):
        try:
            text = self.clipboard_get()
        except tk.TclError:
            return "break"
        vals = [t for t in text.replace("\t", " ").replace(",", " ").split() if t]
        w = event.widget
        try:
            w.delete("sel.first", "sel.last")
        except tk.TclError:
            pass
        before = w.get("1.0", "insert").strip()
        w.insert("insert", (", " if before and not before.endswith(",") else "")
                 + ", ".join(vals))
        return "break"

    def _fill_range(self, key):
        v = self.sens[key]
        try:
            a, b, st = (float(v[k].get()) for k in ("from", "to", "step"))
            vals = value_range(a, b, st)
        except ValueError as e:
            messagebox.showerror("Sensitivity", f"{SENS_PARAMS[key]}: enter a "
                                 f"From, To and Step ({e})")
            return
        v["list"].delete("1.0", "end")
        v["list"].insert("1.0", ", ".join(f"{x:g}" for x in vals))
        v["on"].set(True)

    def _sens_values(self) -> dict:
        out = {}
        for key, v in self.sens.items():
            if not v["on"].get():
                continue
            label = SENS_PARAMS[key]
            try:
                vals = parse_list(v["list"].get("1.0", "end"))
            except ValueError as e:
                raise ValueError(f"{label} sensitivity: {e}") from None
            if not vals:
                raise ValueError(f"{label} sensitivity: the list is empty. Type "
                                 "values or use 'Add range'")
            if len(vals) > 200:
                raise ValueError(f"{label} sensitivity: {len(vals)} values; "
                                 "200 at most")
            out[key] = vals
        return out

    def _cancel(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
            self.status_var.set("Cancelling after the benches in progress...")
            self.cancel_btn.state(["disabled"])

    def _preview(self):
        if self.last_result is None:
            messagebox.showinfo("Plan preview", "Run the dilution first; the preview "
                                "shows the latest run.")
            return
        self.previews = [w for w in self.previews if w.winfo_exists()]
        self.previews.append(PlanPreview(self.winfo_toplevel(), self.last_result))

    def _load_previous(self):
        path = filedialog.askopenfilename(title="Settings of the previous run",
                                          filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            settings = read_settings(path)
            folder = output_folder(settings, path)
            run = LoadedRun(folder)
        except (ValueError, OSError, KeyError) as e:
            _log.error("Could not load the previous run from %s: %s", path, e)
            messagebox.showerror("Load previous run", str(e))
            return
        self._release_result()
        self._apply_settings(settings)
        self.last_result = run
        self._write("-" * 60)
        self._write(f"Loaded the run in {folder} ({len(run.by_bench)} benches, "
                    f"{os.path.basename(run.model)})")
        report = os.path.join(folder, "report.txt")
        if os.path.isfile(report):
            with open(report) as f:
                self._write(f.read())
        self.status_var.set("Previous run loaded")
        if any(f.startswith("sensitivity_") and f.endswith(".csv")
               for f in os.listdir(folder)) and messagebox.askyesno(
                "Load previous run",
                "Write the sensitivity curves and bench CSV again from the "
                "saved models?"):
            try:
                self._write("Sensitivity files written:\n  " +
                            "\n  ".join(rebuild_sensitivity(folder).values()))
            except (ValueError, OSError) as e:
                _log.error("Could not rebuild the sensitivity files: %s", e)
                messagebox.showerror("Sensitivity", str(e))
        self._preview()

    def _release_result(self):
        for w in self.previews:
            try:
                if w.winfo_exists():
                    w.destroy()
            except tk.TclError:
                pass
        self.previews = []
        res, self.last_result = self.last_result, None
        if res is not None:
            res.close()

    def _on_close(self):
        if self.cancel_event is not None:
            self.cancel_event.set()
        try:
            self._release_result()
        except Exception:
            _log.exception("Could not delete the temporary files")
        self.winfo_toplevel().destroy()

    def _history(self):
        HistoryWindow(self.winfo_toplevel())

    def _open_output(self):
        path = self.out_var.get()
        if not path or not os.path.isdir(path):
            messagebox.showinfo("Output folder", "The output folder does not exist yet.")
            return
        _open_path(path)

    def _settings(self) -> dict:
        return {
            "input": self.in_var.get(), "output": self.out_var.get(),
            "columns": {k: v.get() for k, v in self.col_vars.items()},
            "numbers": dict({k: v.get() for k, v in self.num_vars.items()},
                            sample_size=self.sample_var.get()),
            "mode": self.mode_var.get(),
            "ore_rule": self.rule_var.get(),
            "elos_axes": self.elos_axes_var.get(),
            "regions": [dict(name=r.name, priority=r.priority,
                             corners=[list(c) for c in r.corners])
                        for r in self.regions],
            "outside": self.outside_var.get(),
            "audit_fields": self.audit_var.get(),
            "dilute_attr_col": self.dilute_attr_col_var.get(),
            "dilute_attr_values": self.dilute_attr_values_var.get(),
            "merge_subcells": self.merge_var.get(),
            "merge_rules": self._merge_rules(),
            "extra_numeric": self._extras("extra_numeric"),
            "extra_categorical": self._extras("extra_categorical"),
            "extra_metal": self._extras("extra_metal"),
            "options": {"checks": self.check_var.get(),
                        "history": self.hist_var.get(),
                        "dm_only": self.dm_only_var.get(),
                        "sensitivity": {
                            k: {"on": v["on"].get(), "from": v["from"].get(),
                                "to": v["to"].get(), "step": v["step"].get(),
                                "list": v["list"].get("1.0", "end").strip()}
                            for k, v in self.sens.items()}},
        }

    def _save_settings(self):
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._settings(), f, indent=2)
        except OSError as e:
            _log.error("Could not save settings to %s: %s", path, e)
            messagebox.showerror("Settings", f"Could not save the settings:\n{e}")

    def _load_settings(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path) as f:
                settings = json.load(f)
        except OSError as e:
            _log.error("Could not read settings %s: %s", path, e)
            messagebox.showerror("Settings", f"Could not read the settings:\n{e}")
            return
        except json.JSONDecodeError as e:
            _log.error("Settings file %s is not valid JSON: %s", path, e)
            messagebox.showerror("Settings", f"The settings file is not valid JSON:\n{e}")
            return
        if not isinstance(settings, dict):
            _log.error("Settings file %s does not hold a settings object", path)
            messagebox.showerror("Settings", "The file is not a settings file.")
            return
        self._apply_settings(settings)

    def _apply_settings(self, s: dict):
        self.in_var.set(s.get("input", ""))
        self.out_var.set(s.get("output", ""))
        for k, v in s.get("numbers", {}).items():
            if k in self.num_vars and k != "threads":
                self.num_vars[k].set(v)
        if self.in_var.get() and os.path.exists(self.in_var.get()):
            self._load_columns()
        if "threads" in s.get("numbers", {}):
            self.num_vars["threads"].set(s["numbers"]["threads"])
        for k, v in s.get("columns", {}).items():
            if k in self.col_vars:
                self.col_vars[k].set(v)
        self.mode_var.set(s.get("mode", "MII"))
        self.rule_var.set(s.get("ore_rule", "recover"))
        self.elos_axes_var.set(s.get("elos_axes", "xy"))
        self.outside_var.set(s.get("outside", "model"))
        self.audit_var.set(bool(s.get("audit_fields", False)))
        self.dilute_attr_col_var.set(s.get("dilute_attr_col", NONE))
        self.dilute_attr_values_var.set(s.get("dilute_attr_values", ""))
        self.merge_var.set(bool(s.get("merge_subcells", False)))
        self.sample_var.set(s.get("numbers", {}).get("sample_size", ""))
        try:
            self.regions = as_regions(s.get("regions", []))
        except (ValueError, KeyError, TypeError) as e:
            _log.error("Ignored the saved regions: %s", e)
            self.regions = []
        self._refresh_regions()
        self._set_extras("extra_numeric", s.get("extra_numeric", []))
        self._set_extras("extra_categorical", s.get("extra_categorical", []))
        self._set_extras("extra_metal", s.get("extra_metal", []))
        for name, rule in (s.get("merge_rules") or {}).items():
            label = MERGE_RULES.get(rule)
            if label:
                self.merge_choice.setdefault(name, tk.StringVar()).set(label)
        self._refresh_merge_attrs()
        o = s.get("options", {})
        self.check_var.set(o.get("checks", True))
        self.hist_var.set(o.get("history", True))
        self.dm_only_var.set(bool(o.get("dm_only", False)))
        sens = o.get("sensitivity", {})
        if not isinstance(sens, dict):
            sens = {"cutoff": {"on": bool(sens), "list": o.get("sens_cutoffs", "")},
                    "smu_x": {"on": bool(sens and o.get("sens_smu_x")),
                              "list": o.get("sens_smu_x", "")}}
        for k, v in self.sens.items():
            cfg = sens.get(k, {})
            v["on"].set(bool(cfg.get("on", False)))
            for part in ("from", "to", "step"):
                if part in cfg:
                    v[part].set(cfg[part])
            v["list"].delete("1.0", "end")
            v["list"].insert("1.0", cfg.get("list", ""))
            if cfg.get("on") and not cfg.get("list") and cfg.get("step"):
                try:
                    v["list"].insert("1.0", ", ".join(f"{x:g}" for x in value_range(
                        float(cfg["from"]), float(cfg["to"]), float(cfg["step"]))))
                except (KeyError, ValueError) as e:
                    _log.debug("Ignored the saved %s range: %s", k, e)

    def _write(self, text: str):
        for line in text.split("\n"):
            word = line.split(":")[0].split(" ")[0]
            if word in T.STATUS and (line.startswith(word + " ") or
                                     line.startswith(word + ":")):
                rest = line[len(word):]
                if rest.startswith("   "):
                    rest = rest[2:]
                self.log.insert("end", f"{T.STATUS[word]['icon']} {word}", word)
                self.log.insert("end", rest + "\n")
            else:
                self.log.insert("end", line + "\n")
        self.log.see("end")

    def _drain(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg is None:
                    self.run_btn.state(["!disabled"])
                    self.cancel_btn.state(["disabled"])
                elif isinstance(msg, tuple) and msg[0] in ("progress", "sens"):
                    _, done, total = msg
                    self.bar["maximum"] = total
                    self.bar["value"] = done
                    el = time.time() - self.t0
                    what = "Bench" if msg[0] == "progress" else "Sensitivity run"
                    self.status_var.set(f"{what} {done} of {total}, {el:,.0f} s elapsed")
                elif isinstance(msg, tuple) and msg[0] == "status":
                    self.status_var.set(msg[1])
                elif isinstance(msg, tuple) and msg[0] == "result":
                    self.last_result = msg[1]
                elif isinstance(msg, tuple) and msg[0] == "preview":
                    RegionPreview(self.winfo_toplevel(), *msg[1:])
                else:
                    self._write(msg)
        except queue.Empty:
            pass
        self.after(100, self._drain)

def _legend(parent, fill: str, outline: str, dash: tuple, text: str,
            line: bool = True):
    sw = tk.Canvas(parent, width=26, height=16, background=T.CHART_BG,
                   highlightthickness=0)
    if line:
        sw.create_rectangle(3, 3, 23, 13, outline=outline, width=3, dash=dash)
    else:
        sw.create_rectangle(2, 2, 24, 14, fill=fill, outline=outline or fill)
    sw.pack(side="left")
    ttk.Label(parent, text=text).pack(side="left", padx=(4, 14))

def _heading(name: str) -> str:
    keep = {"AU", "SMU", "OZ", "X", "Y", "Z", "ID", "MII", "MI"}
    swap = {"CUTOFF": "cut-off", "PCT": "%"}
    words = name.replace("_", " ").split()
    out = [w if w in keep else swap.get(w, w.lower()) for w in words]
    if out and out[0] not in keep:
        out[0] = out[0].capitalize()
    return " ".join(out)

def _open_path(path: str):
    try:
        if sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
    except OSError as e:
        _log.error("Could not open %s: %s", path, e)
        messagebox.showerror("Open", f"Could not open {path}:\n{e}")

class PlanPreview(tk.Toplevel):

    MAX_CELLS = 60000

    def __init__(self, master, res):
        super().__init__(master)
        T.toplevel(self)
        self.title("Plan preview")
        self.res = res
        p = res.params
        self.cutoff = p["cutoff"]

        grid = res.grid
        self.extent = (grid["xlo"], grid["xhi"], grid["ylo"], grid["yhi"])
        self.def_name = def_col(p)
        self.sz = p["smu_z"]
        self.regions = order_regions(as_regions(p.get("regions") or []))
        oxides = sorted(res.ore_spool.oxides)
        self.ox_style = {o: (T.ORE_OUTLINES[i % len(T.ORE_OUTLINES)], ())
                         for i, o in enumerate(oxides)}

        zs = res.by_bench.ZC.tolist() if len(res.by_bench) else []
        self.benches = [f"{z:g}" for z in sorted(zs, reverse=True)]
        self.bench_rows = {f"{r.ZC:g}": r for r in res.by_bench.itertuples()} \
            if len(res.by_bench) else {}

        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(fill="x")
        ttk.Label(top, text="Bench centre Z (m)").pack(side="left")
        self.bench_var = tk.StringVar()
        cb = ttk.Combobox(top, textvariable=self.bench_var, values=self.benches,
                          state="readonly", width=10)
        cb.pack(side="left", padx=8)
        cb.bind("<<ComboboxSelected>>", lambda e: self.draw())
        ttk.Button(top, text="Up", width=5,
                   command=lambda: self._step(-1)).pack(side="left")
        ttk.Button(top, text="Down", width=6,
                   command=lambda: self._step(1)).pack(side="left", padx=4)
        ttk.Button(top, text="Fit", width=5, command=self.draw).pack(side="left", padx=(8, 0))
        ttk.Label(top, text="Page Up/Down change bench, F fits", style="Help.TLabel").pack(
            side="right")
        self.info_var = tk.StringVar()
        ttk.Label(self, textvariable=self.info_var, padding=(8, 0, 8, 4),
                  font=T.fonts.bold).pack(fill="x")
        self.bind("<Prior>", lambda e: self._step(-1))
        self.bind("<Next>", lambda e: self._step(1))
        self.bind("f", lambda e: self.draw())

        self.canvas = tk.Canvas(self, width=860, height=560, background=T.CHART_BG)
        self.canvas.pack(fill="both", expand=True, padx=8)
        leg = ttk.Frame(self, padding=(8, 6))
        leg.pack(fill="x")
        _legend(leg, T.CHART_ORE_CELL, T.CHART_ORE_CELL_EDGE, (),
                f"cell {self.def_name} \u2265 {self.cutoff:g}", line=False)
        _legend(leg, T.CHART_MODEL, "", (), "cell below cut-off", line=False)
        for o, (colour, dash) in self.ox_style.items():
            _legend(leg, "", colour, dash, f"ore block {o}")
        self.hover_var = tk.StringVar(value="Hover over a block for details. "
                                      "Wheel zooms, drag pans.")
        ttk.Label(self, textvariable=self.hover_var, padding=(8, 0, 8, 8),
                  font=T.fonts.mono, style="Help.TLabel").pack(fill="x")

        self.items: dict[int, str] = {}
        c = self.canvas
        c.bind("<Motion>", self._hover)
        c.bind("<ButtonPress-1>", lambda e: c.scan_mark(e.x, e.y))
        c.bind("<B1-Motion>", lambda e: c.scan_dragto(e.x, e.y, gain=1))
        c.bind("<MouseWheel>", lambda e: self._zoom(e, 1.2 if e.delta > 0 else 1 / 1.2))
        c.bind("<Button-4>", lambda e: self._zoom(e, 1.2))
        c.bind("<Button-5>", lambda e: self._zoom(e, 1 / 1.2))
        if self.benches:
            best = (res.by_bench.sort_values("DILUTED_TONNES", ascending=False)
                    .ZC.iloc[0])
            self.bench_var.set(f"{best:g}")
        self.after(50, self.draw)

    def _step(self, d):
        if not self.benches:
            return
        i = self.benches.index(self.bench_var.get()) if self.bench_var.get() in self.benches else 0
        self.bench_var.set(self.benches[max(0, min(len(self.benches) - 1, i + d))])
        self.draw()

    def _zoom(self, e, f):
        c = self.canvas
        c.scale("all", c.canvasx(e.x), c.canvasy(e.y), f, f)

    def draw(self):
        c = self.canvas
        c.delete("all")
        self.items.clear()
        c.xview_moveto(0)
        c.yview_moveto(0)
        if not self.bench_var.get():
            c.create_text(20, 20, anchor="nw", text="No benches in this run",
                          fill=T.CHART_TEXT, font=T.fonts.body)
            return
        zc = float(self.bench_var.get())
        cells = self.res.bench_cells(zc)
        self.X, self.Y, self.Z = cells["X"], cells["Y"], cells["Z"]
        self.XS, self.YS = cells["xs"], cells["ys"]
        self.AUD, self.ORE = cells["def"], cells["ore"]
        sel = np.arange(len(self.X))
        ob = self.res.ore_spool.bench(zc)

        x0, x1, y0, y1 = self.extent
        W = max(c.winfo_width(), 300) - 20
        H = max(c.winfo_height(), 200) - 20
        s = min(W / max(x1 - x0, 1e-9), H / max(y1 - y0, 1e-9))

        def px(x):
            return 10 + (x - x0) * s

        def py(y):
            return 10 + (y1 - y) * s

        note = ""
        if len(sel) > self.MAX_CELLS:
            note = f" (showing {self.MAX_CELLS:,} of {len(sel):,} cells)"
            sel = sel[np.argsort(-self.AUD[sel])[:self.MAX_CELLS]]
        for i in sel:
            above = self.AUD[i] >= self.cutoff
            it = c.create_rectangle(
                px(self.X[i] - self.XS[i] / 2), py(self.Y[i] + self.YS[i] / 2),
                px(self.X[i] + self.XS[i] / 2), py(self.Y[i] - self.YS[i] / 2),
                fill=T.CHART_ORE_CELL if above else T.CHART_MODEL,
                outline=T.CHART_ORE_CELL_EDGE if above else "")
            self.items[it] = (f"Cell X {self.X[i]:g} Y {self.Y[i]:g} Z {self.Z[i]:g}  "
                              f"{self.def_name} {self.AUD[i]:.3f}  in ore {100 * self.ORE[i]:.0f}%")
        for i, rg in enumerate(self.regions):
            pts = [v for x, y in rg.polygon() for v in (px(x), py(y))]

            c.create_polygon(*pts, outline=T.CHART_REGION, fill="", width=1,
                             dash=T.dash(i + 1))
            cx, cy = rg.to_world(rg.length / 2, rg.width * 0.7)
            c.create_text(px(float(cx)), py(float(cy)), text=f"{rg.name} (P{rg.priority})",
                          fill=T.CHART_TEXT, font=T.fonts.bold)
        for r in ob.itertuples(index=False):
            col, dash = self.ox_style.get(str(r.OXIDE), (T.CHART_TEXT, ()))

            pts = [v for x, y, _ in block_outline(r)[:-1] for v in (px(x), py(y))]
            it = c.create_polygon(*pts, outline=col, width=3, dash=dash, fill="")
            self.items[it] = (f"Ore block {r.BLOCK_ID} (zone {r.ZONE_ID})  X "
                              f"{r.X_FROM:g}-{r.X_TO:g}  {r.TONNES:,.0f} t @ "
                              f"{r.AU:.3f}  {self.def_name} "
                              f"{getattr(r, self.def_name):.3f}  {r.OXIDE} "
                              f"{r.OXIDE_PCT:.0f}%")
        b = self.bench_rows.get(self.bench_var.get())
        if b is not None:
            self.info_var.set(f"{b.ORE_BLOCKS:,} ore blocks, {b.DILUTED_TONNES:,.0f} t "
                              f"@ {b.DILUTED_AU:.2f} g/t diluted (in situ "
                              f"{b.UNDILUTED_TONNES:,.0f} t @ {b.UNDILUTED_AU:.2f} g/t)"
                              f"{note}")

    def _hover(self, e):
        c = self.canvas
        x, y = c.canvasx(e.x), c.canvasy(e.y)
        hits = [i for i in c.find_overlapping(x, y, x, y) if i in self.items]
        if hits:
            ore = [i for i in hits if self.items[i].startswith("Ore")]
            self.hover_var.set(self.items[(ore or hits)[-1]])

class RegionPreview(tk.Toplevel):

    def __init__(self, master, plan: dict, regions: list, p: DilutionParams):
        super().__init__(master)
        T.toplevel(self)
        self.title("Region preview")
        self.plan, self.regions, self.p = plan, regions, p
        top = ttk.Frame(self, padding=(8, 8, 8, 4))
        top.pack(fill="x")
        ttk.Label(top, text=f"Ore = {p.grade_col}_{p.mode} \u2265 {p.cutoff:g} on any "
                  "bench. Arrows show the dilution direction.").pack(side="left")
        ttk.Button(top, text="Save PDF...", command=self._save).pack(side="right")
        self.canvas = tk.Canvas(self, width=860, height=520, background=T.CHART_BG)
        self.canvas.pack(fill="both", expand=True, padx=8)
        leg = ttk.Frame(self, padding=(8, 6))
        leg.pack(fill="x")
        _legend(leg, T.CHART_MODEL, "", (), "below cut-off", line=False)
        for i, r in enumerate(regions):
            _legend(leg, region_colour(i), region_colour(i), (), f"ore in {r.name}",
                    line=False)
        _legend(leg, T.CHART_OUTSIDE, T.CHART_OUTSIDE, (), "ore outside every region",
                line=False)
        cols = ("REGION", "PRIORITY", "AZIMUTH", "ORE_TONNES", "ORE_GRADE", "NOTE")
        tv = ttk.Treeview(self, columns=cols, show="headings",
                          height=min(len(regions) + 1, 8))
        for c in cols:
            tv.heading(c, text=_heading(c))
            tv.column(c, width=320 if c == "NOTE" else 100,
                      anchor="w" if c in ("REGION", "NOTE") else "e")
        for r in region_table(plan, regions, p.outside).itertuples(index=False):
            tv.insert("", "end", values=(
                r.REGION, "" if r.PRIORITY is None or r.PRIORITY != r.PRIORITY
                else int(r.PRIORITY),
                "" if r.AZIMUTH is None or r.AZIMUTH != r.AZIMUTH else f"{r.AZIMUTH:05.1f}",
                f"{r.ORE_TONNES:,.0f}", f"{r.ORE_GRADE:,.2f}", r.NOTE))
        tv.pack(fill="x", padx=8, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda e: self.draw())

    def draw(self):
        c, plan = self.canvas, self.plan
        c.delete("all")
        xlo, ylo, res = plan["xlo"], plan["ylo"], plan["res"]
        x0, x1 = xlo, xlo + plan["nx"] * res
        y0, y1 = ylo, ylo + plan["ny"] * res
        for r in self.regions:
            a, b, cc, d = r.bbox()
            x0, x1, y0, y1 = min(x0, a), max(x1, b), min(y0, cc), max(y1, d)
        W = max(c.winfo_width(), 300) - 20
        H = max(c.winfo_height(), 200) - 20
        s = min(W / max(x1 - x0, 1e-9), H / max(y1 - y0, 1e-9))

        def px(x):
            return 10 + (x - x0) * s

        def py(y):
            return 10 + (y1 - y) * s

        for _, colour, runs_ in plan_layers(plan, self.regions, model_colour=T.CHART_MODEL):
            for j, a, b in runs_:
                c.create_rectangle(px(xlo + a * res), py(ylo + (j + 1) * res),
                                   px(xlo + b * res), py(ylo + j * res),
                                   fill=colour, outline="")
        for i, r in enumerate(self.regions):
            pts = [v for x, y in r.polygon() for v in (px(x), py(y))]
            c.create_polygon(*pts, outline=region_colour(i), fill="", width=2,
                             dash=T.dash(i))
            (ax, ay), (bx, by) = arrow(r)
            c.create_line(px(ax), py(ay), px(bx), py(by), arrow="last", width=2,
                          fill=T.CHART_TEXT)
            cx, cy = r.to_world(r.length / 2, r.width * 0.7)
            c.create_text(px(float(cx)), py(float(cy)), text=f"{r.name} (P{r.priority})",
                          fill=T.CHART_TEXT, font=T.fonts.bold)

    def _save(self):
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".pdf",
                                            initialfile="regions_preview.pdf",
                                            filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        try:
            write_pdf(path, self.plan, self.regions, self.p)
        except (OSError, RuntimeError) as e:
            _log.error("Could not write the region preview %s: %s", path, e)
            messagebox.showerror("Region preview", f"Could not write the PDF:\n{e}",
                                 parent=self)
            return
        _open_path(path)

class HistoryWindow(tk.Toplevel):
    COLUMNS = ["RUN_TIME", "USER", "INPUT", "MODE", "CUTOFF", "SMU_X", "SMU_Y",
               "SMU_Z", "STEP_X", "STEP_Y", "DILUTED_TONNES", "DILUTED_AU", "DILUTED_OZ",
               "DILUTION_PCT", "ORE_LOSS_PCT", "SECONDS", "OUTPUT"]

    DECIMALS = {"DILUTED_TONNES": 0, "DILUTED_OZ": 0, "SECONDS": 0, "DILUTION_PCT": 1,
                "ORE_LOSS_PCT": 1}

    def __init__(self, master):
        super().__init__(master)
        T.toplevel(self)
        self.title("Run history")
        h = read_history()
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text=f"{len(h)} runs in {HISTORY_FILE}").pack(side="left")
        ttk.Button(top, text="Open history file", command=self._open).pack(side="right")
        fr = ttk.Frame(self, padding=(8, 0, 8, 8))
        fr.pack(fill="both", expand=True)
        cols = [c for c in self.COLUMNS if c in h.columns]
        tv = ttk.Treeview(fr, columns=cols, show="headings", height=18)
        for c in cols:
            tv.heading(c, text=_heading(c))
            fit = T.fonts.bold.measure(_heading(c)) + 24
            tv.column(c, width=max(fit, 160 if c in ("INPUT", "OUTPUT", "RUN_TIME") else 90),
                      stretch=False,
                      anchor="w" if c in ("INPUT", "OUTPUT", "USER", "RUN_TIME", "MODE") else "e")
        for r in h.iloc[::-1].itertuples(index=False):
            vals = []
            for c in cols:
                v = getattr(r, c)
                if isinstance(v, float):
                    v = "" if np.isnan(v) else f"{v:,.{self.DECIMALS.get(c, 2)}f}"
                if c in ("INPUT", "OUTPUT"):
                    v = os.path.basename(str(v)) or str(v)
                vals.append(v)
            tv.insert("", "end", values=vals)
        ys = ttk.Scrollbar(fr, orient="vertical", command=tv.yview)
        xs = ttk.Scrollbar(fr, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        tv.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        fr.columnconfigure(0, weight=1)
        fr.rowconfigure(0, weight=1)

    def _open(self):
        if os.path.exists(HISTORY_FILE):
            _open_path(HISTORY_FILE)
        else:
            messagebox.showinfo("Run history", "No runs have been logged yet.", parent=self)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

def set_window_icon(root: tk.Tk) -> None:
    try:
        logo = tk.PhotoImage(file=os.path.join(ASSETS, "logo.png"))
        root.iconphoto(True, logo)
        root._logo = logo
        if sys.platform == "win32":

            root.iconbitmap(default=os.path.join(ASSETS, "logo.ico"))
    except tk.TclError as e:
        _log.warning("Could not load the window icon: %s", e)

APP_ID = "OpenPitDilution2D"

def set_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except (AttributeError, OSError) as e:
        _log.debug("Could not set the taskbar app id: %s", e)

def main(on_ready=None):
    set_app_id()
    root = tk.Tk()
    set_window_icon(root)
    DilutionApp(root)
    if on_ready is not None:
        root.update_idletasks()
        root.after_idle(on_ready)
    root.mainloop()
