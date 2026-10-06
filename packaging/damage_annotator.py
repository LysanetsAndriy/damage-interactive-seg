"""Entry point of the packaged app (PyInstaller)."""
import sys

from dmgseg.app.gui import main

if __name__ == "__main__":
    sys.exit(main())
