"""Make SAM 2 importable without pip-installing it (its pip package pins torch,
which molab's preinstalled torch must not be replaced by)."""
import subprocess
import sys
from pathlib import Path

SAM2_URL = "https://github.com/facebookresearch/sam2.git"


def ensure_sam2(target=Path.home() / "sam2"):
    try:
        import sam2  # noqa: F401
        return "already importable"
    except ImportError:
        pass
    target = Path(target)
    if not (target / "sam2").exists():
        subprocess.run(["git", "clone", "--depth", "1", SAM2_URL, str(target)], check=True)
    sys.path.insert(0, str(target))
    import sam2  # noqa: F401
    return f"from {target}"
