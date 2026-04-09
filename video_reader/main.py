"""Application entry point for the video frame browser."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui_mainwindow import MainWindow


def main() -> int:
    """Create the Qt application and run the main window."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
