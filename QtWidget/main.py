#!/usr/bin/env python3
from abc import ABCMeta, abstractmethod
from qtpy.QtWidgets import QWidget, QApplication
import sys
from WebGPUWidget import WebGPUWidget
from qtpy.QtCore import Qt


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of AbstractWebGPUWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def initializeWebGPU(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        print("initializeWebGPU")

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        print("paintWebGPU")
        self.render_text(10, 20, "PaintGPU Test", size=20, colour=Qt.yellow)

    def resizeWebGPU(self, width: int, height: int) -> None:
        """
        Resize the WebGPU context.

        This method handles resizing of the WebGPU context for the scene.

        Args:
            width (int): The new width of the widget.
            height (int): The new height of the widget.
        """
        print(f"resizeWebGPU {width} {height}")


app = QApplication(sys.argv)
win = WebGPUScene()
win.show()
sys.exit(app.exec())
