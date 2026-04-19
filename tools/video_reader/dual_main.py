"""Application entry point for the dual-video frame browser."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.video_reader.dual_mainwindow import DualMainWindow


def main() -> int:
    """Create the Qt application and run the dual-video main window."""
    app = QApplication(sys.argv)
    window = DualMainWindow()
    window.resize(1600, 700)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
