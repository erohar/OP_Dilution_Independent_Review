from __future__ import annotations

import logging
import sys

_log = logging.getLogger(__name__)

YELLOW = "#DCA91F"
ORANGE = "#C37529"
RED = "#B04224"
BROWN = "#743929"
BLACK = "#302E2C"
GOLD = "#BA9850"
BLUE = "#6698C6"
TEAL = "#008080"
GREEN = "#A9A534"
GREY = "#96929D"

BASE = "#121212"
SURFACE = "#1E1E1E"
RAISED = BLACK
BORDER = "#3D3A37"
TEXT = "#ECEAE7"
TEXT_2 = "#B0ADA8"
DISABLED = "#7A7772"

ACCENT = YELLOW
ON_ACCENT = BLACK
HEADING = ORANGE
SELECT_BG = BLUE
SELECT_FG = BASE
INPUT_OUTLINE = GREY
FOCUS = YELLOW

STATUS = {
    "PASS": {"icon": "✓", "fg": GREEN, "bg": None},
    "WARN": {"icon": "▲", "fg": ORANGE, "bg": None},
    "FAIL": {"icon": "✕", "fg": TEXT, "bg": RED},
    "ERROR": {"icon": "✕", "fg": TEXT, "bg": RED},
    "INFO": {"icon": "ℹ", "fg": BLUE, "bg": None},
}

CHART_BG = SURFACE
CHART_GRID = BORDER
CHART_TEXT = TEXT
CHART_MODEL = RAISED
CHART_ORE_CELL = BROWN
CHART_ORE_CELL_EDGE = ORANGE
CHART_OUTSIDE = GREY
CHART_REGION = GREY

SERIES = [YELLOW, ORANGE, BLUE, TEAL, GREEN, GOLD]

ORE_OUTLINES = [RED, YELLOW, GREEN, GOLD, BLUE, GREY, TEAL]

DASHES = [(), (8, 4), (2, 3), (10, 3, 2, 3), (5, 5), (1, 2), (12, 4, 2, 4, 2, 4)]

PRINT_MODEL = "#E4E4E4"
PRINT_FRAME = "#999999"

MIN_PT = 10
BODY_PT = 10
HEADING_PT = 11
TITLE_PT = 13

def series(i: int) -> str:
    return SERIES[i % len(SERIES)]

def dash(i: int) -> tuple:
    return DASHES[i % len(DASHES)]

def _family(root, wanted: list[str]) -> str:
    from tkinter import font as tkfont
    have = {f.lower(): f for f in tkfont.families(root)}
    for w in wanted:
        if w.lower() in have:
            return have[w.lower()]
    return wanted[-1]

class Fonts:
    body = heading = title = bold = mono = None

fonts = Fonts()

def _make_fonts(root):
    import tkinter as tk
    from tkinter import font as tkfont

    sans = _family(root, ["Arial", "Liberation Sans", "Helvetica"])
    head = _family(root, ["Gotham", sans])
    mono = _family(root, ["Consolas", "Liberation Mono", "Courier New", "Courier"])
    fonts.body = tkfont.Font(root, family=sans, size=BODY_PT)
    fonts.bold = tkfont.Font(root, family=sans, size=BODY_PT, weight="bold")
    fonts.heading = tkfont.Font(root, family=head, size=HEADING_PT, weight="bold")
    fonts.title = tkfont.Font(root, family=head, size=TITLE_PT,
                              weight="normal" if head == sans else "bold")
    fonts.mono = tkfont.Font(root, family=mono, size=BODY_PT)
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
                 "TkCaptionFont", "TkTooltipFont", "TkIconFont", "TkSmallCaptionFont"):
        try:
            tkfont.nametofont(name, root).configure(family=sans, size=BODY_PT)
        except tk.TclError:
            pass
    try:
        tkfont.nametofont("TkFixedFont", root).configure(family=mono, size=BODY_PT)
    except tk.TclError:
        pass

def _options(root):
    o = root.option_add
    for pat, val in (("*Background", SURFACE), ("*background", SURFACE),
                     ("*Foreground", TEXT), ("*foreground", TEXT),
                     ("*selectBackground", SELECT_BG), ("*selectForeground", SELECT_FG),
                     ("*insertBackground", TEXT), ("*highlightColor", FOCUS),
                     ("*highlightBackground", SURFACE), ("*activeBackground", RAISED),
                     ("*activeForeground", TEXT), ("*disabledForeground", DISABLED),
                     ("*troughColor", SURFACE),
                     ("*Listbox.background", RAISED), ("*Listbox.relief", "flat"),
                     ("*Listbox.highlightThickness", 1),
                     ("*Listbox.highlightBackground", INPUT_OUTLINE),
                     ("*Text.background", RAISED), ("*Text.relief", "flat"),
                     ("*Text.highlightThickness", 1),
                     ("*Text.highlightBackground", INPUT_OUTLINE),
                     ("*Canvas.background", CHART_BG), ("*Canvas.highlightThickness", 0),
                     ("*TCombobox*Listbox.background", RAISED),
                     ("*TCombobox*Listbox.foreground", TEXT),
                     ("*TCombobox*Listbox.selectBackground", SELECT_BG),
                     ("*TCombobox*Listbox.selectForeground", SELECT_FG),
                     ("*TCombobox*Listbox.font", "TkDefaultFont"),
                     ("*Menu.background", RAISED), ("*Menu.foreground", TEXT)):
        o(pat, val)

def _style(root):
    import tkinter as tk
    from tkinter import ttk
    s = ttk.Style(root)
    try:
        s.theme_use("clam")
    except tk.TclError as e:
        _log.warning("The clam theme is not available: %s", e)
    s.configure(".", background=SURFACE, foreground=TEXT, fieldbackground=RAISED,
                bordercolor=BORDER, lightcolor=SURFACE, darkcolor=SURFACE,
                troughcolor=SURFACE, focuscolor=FOCUS, selectbackground=SELECT_BG,
                selectforeground=SELECT_FG, insertcolor=TEXT, arrowcolor=TEXT,
                font=fonts.body)
    s.map(".", foreground=[("disabled", DISABLED)])

    s.configure("Base.TFrame", background=BASE)
    s.configure("Base.TLabel", background=BASE, foreground=TEXT)
    s.configure("Help.TLabel", foreground=TEXT_2)
    s.configure("Heading.TLabel", foreground=HEADING, font=fonts.heading)
    s.configure("Banner.TFrame", background=RAISED)
    s.configure("Banner.TLabel", background=RAISED, foreground=TEXT)
    s.configure("BannerKey.TLabel", background=RAISED, foreground=ACCENT,
                font=fonts.bold)
    s.configure("BannerMode.TLabel", background=RAISED, foreground=TEXT,
                font=fonts.bold)
    for key, st in STATUS.items():
        s.configure(f"{key.title()}.TLabel", foreground=st["fg"],
                    background=st["bg"] or SURFACE, font=fonts.bold)

    s.configure("TLabelframe", background=SURFACE, bordercolor=BORDER,
                lightcolor=BORDER, darkcolor=BORDER, relief="solid", borderwidth=1)
    s.configure("TLabelframe.Label", background=SURFACE, foreground=HEADING,
                font=fonts.heading)

    s.configure("TNotebook", background=BASE, bordercolor=BORDER, lightcolor=BASE,
                darkcolor=BASE, tabmargins=(0, 4, 0, 0))
    s.configure("TNotebook.Tab", background=SURFACE, foreground=TEXT_2,
                bordercolor=BORDER, lightcolor=SURFACE, darkcolor=SURFACE,
                padding=(12, 6), font=fonts.body)
    s.map("TNotebook.Tab",
          background=[("selected", RAISED), ("active", RAISED)],
          foreground=[("selected", ACCENT), ("active", TEXT)],
          lightcolor=[("selected", ACCENT)],
          font=[("selected", fonts.bold)])

    s.configure("TButton", background=RAISED, foreground=TEXT, bordercolor=INPUT_OUTLINE,
                lightcolor=RAISED, darkcolor=RAISED, padding=(10, 4), focusthickness=2)
    s.map("TButton",
          background=[("disabled", SURFACE), ("pressed", BORDER), ("active", BORDER)],
          bordercolor=[("focus", FOCUS), ("disabled", BORDER)],
          lightcolor=[("pressed", BORDER), ("active", BORDER)],
          darkcolor=[("pressed", BORDER), ("active", BORDER)],
          foreground=[("disabled", DISABLED)])
    s.configure("Small.TButton", padding=(8, 1))
    s.configure("Accent.TButton", background=ACCENT, foreground=ON_ACCENT,
                bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT, font=fonts.bold)
    s.map("Accent.TButton",

          background=[("disabled", SURFACE), ("pressed", GOLD), ("active", GOLD)],
          lightcolor=[("pressed", GOLD), ("active", GOLD)],
          darkcolor=[("pressed", GOLD), ("active", GOLD)],
          bordercolor=[("focus", TEXT), ("disabled", BORDER)],
          foreground=[("disabled", DISABLED)])

    for w in ("TEntry", "TCombobox", "TSpinbox"):
        s.configure(w, fieldbackground=RAISED, foreground=TEXT, bordercolor=INPUT_OUTLINE,
                    lightcolor=RAISED, darkcolor=RAISED, background=RAISED,
                    arrowcolor=TEXT, padding=(4, 2))
        s.map(w, bordercolor=[("focus", FOCUS), ("hover", TEXT)],
              lightcolor=[("focus", FOCUS)],
              background=[("pressed", BORDER), ("active", BORDER)],
              arrowcolor=[("disabled", DISABLED)],
              fieldbackground=[("readonly", RAISED), ("disabled", SURFACE)],
              foreground=[("disabled", DISABLED)],
              selectbackground=[("!focus", RAISED)], selectforeground=[("!focus", TEXT)])

    for w in ("TCheckbutton", "TRadiobutton"):
        s.configure(w, background=SURFACE, foreground=TEXT, indicatorbackground=RAISED,
                    indicatorforeground=ON_ACCENT, indicatorcolor=RAISED,
                    upperbordercolor=INPUT_OUTLINE, lowerbordercolor=INPUT_OUTLINE,
                    focuscolor=FOCUS, padding=(2, 2))
        s.map(w, background=[("active", SURFACE)],
              indicatorcolor=[("selected", ACCENT), ("pressed", BORDER)],
              indicatorbackground=[("selected", ACCENT), ("pressed", BORDER)],
              foreground=[("disabled", DISABLED)])

    s.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=RAISED,
                bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
    for o in ("Vertical", "Horizontal"):
        s.configure(f"{o}.TScrollbar", background=RAISED, troughcolor=SURFACE,
                    bordercolor=BORDER, lightcolor=RAISED, darkcolor=RAISED,
                    arrowcolor=TEXT_2, gripcount=0)
        s.map(f"{o}.TScrollbar", background=[("active", BORDER)])

    s.configure("Treeview", background=SURFACE, fieldbackground=SURFACE, foreground=TEXT,
                bordercolor=BORDER, lightcolor=SURFACE, darkcolor=SURFACE,
                rowheight=fonts.body.metrics("linespace") + 8)
    s.map("Treeview", background=[("selected", SELECT_BG)],
          foreground=[("selected", SELECT_FG)])
    s.configure("Treeview.Heading", background=RAISED, foreground=TEXT, font=fonts.bold,
                bordercolor=BORDER, lightcolor=RAISED, darkcolor=RAISED, padding=(6, 4))
    s.map("Treeview.Heading", background=[("active", BORDER)])
    s.configure("TSeparator", background=BORDER)

def apply(root) -> None:
    import tkinter as tk
    _make_fonts(root)
    _options(root)
    _style(root)
    try:
        root.configure(background=BASE)
    except tk.TclError:
        pass
    if sys.platform == "win32":
        _dark_title_bar(root)

def toplevel(win) -> None:
    win.configure(background=BASE)
    if sys.platform == "win32":
        _dark_title_bar(win)

def _dark_title_bar(win) -> None:
    import tkinter as tk
    try:
        import ctypes
        win.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
        value = ctypes.c_int(1)
        for attr in (20, 19):
            if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)) == 0:
                break
    except (AttributeError, OSError, tk.TclError) as e:
        _log.debug("Dark title bar not available: %s", e)
