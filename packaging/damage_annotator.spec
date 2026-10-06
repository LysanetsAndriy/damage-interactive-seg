# PyInstaller recipe for the macOS app (ONNX backend: no PyTorch inside).
#   cd packaging && ../.venv/bin/pyinstaller --noconfirm damage_annotator.spec
from pathlib import Path

ROOT = Path(SPECPATH).parent
ONNX = ROOT / "artifacts" / "onnx"
models = [(str(f), "models") for f in sorted(ONNX.glob("*.onnx"))] + [(str(ONNX / "meta.json"), "models")]

a = Analysis(
    ["damage_annotator.py"],
    pathex=[str(ROOT / "src")],
    datas=models,
    hiddenimports=["dmgseg.app.selftest", "dmgseg.onnx.runtime", "scipy.ndimage"],
    # the PyTorch path is never used by the packaged app (lazy imports only)
    excludes=["torch", "torchvision", "torchaudio", "timm", "sam2", "hydra", "iopath", "transformers",
              "albumentations", "huggingface_hub", "matplotlib", "pandas", "sklearn", "marimo", "IPython",
              "tkinter", "onnx", "pytest", "sympy", "triton", "networkx"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Damage Annotator", console=False,
          argv_emulation=False, icon=str(Path(SPECPATH) / "build" / "icon.icns"))
coll = COLLECT(exe, a.binaries, a.datas, name="Damage Annotator")
app = BUNDLE(
    coll, name="Damage Annotator.app", icon=str(Path(SPECPATH) / "build" / "icon.icns"),
    bundle_identifier="ua.knu.damage-annotator",
    info_plist={"CFBundleShortVersionString": "1.0.0", "CFBundleVersion": "1.0.0",
                "NSHighResolutionCapable": True, "LSMinimumSystemVersion": "10.15",
                "NSHumanReadableCopyright": "Andriy Lysanets, 2026"},
)
