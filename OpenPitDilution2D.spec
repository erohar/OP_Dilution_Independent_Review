import sys

from PyInstaller.utils.hooks import collect_data_files

APP_NAME = "OpenPitDilution2D"

a = Analysis(
    ["run_dilution.py"],
    pathex=[SPECPATH],
    binaries=[],

    datas=collect_data_files("reportlab") + [
        ("dilution/assets/logo.png", "dilution/assets"),
        ("dilution/assets/logo.ico", "dilution/assets"),
    ],

    hiddenimports=[
        "dilution.gui", "dilution.preview", "dilution.theme",
        "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox",
        "tkinter.font", "pyi_splash",
        "reportlab.pdfgen.canvas", "reportlab.lib.pagesizes", "reportlab.lib.colors",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    excludes=["matplotlib", "scipy", "IPython", "jupyter", "notebook", "pytest",
              "PyQt5", "PyQt6", "PySide2", "PySide6", "sphinx", "pyarrow"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

splash_parts = []
if sys.platform != "darwin":
    splash = Splash(
        "dilution/assets/splash.png",
        binaries=a.binaries,
        datas=a.datas,
        text_pos=None,
        minify_script=True,
        always_on_top=False,
    )
    splash_parts = [splash, splash.binaries]

exe = EXE(
    pyz,
    a.scripts,
    *splash_parts,
    a.binaries,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,

    upx=False,
    runtime_tmpdir=None,

    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,

    icon="dilution/assets/logo.ico",
)
