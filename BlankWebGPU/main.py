#!/usr/bin/env -S uv run --script
import sys

from qtpy.QtWidgets import QApplication
from WebGPUWidget import WebGPUWidget


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of AbstractWebGPUWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self, width=1024, height=720):
        super().__init__(width, height)

    def initializeWebGPU(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        super().initializeWebGPU()
        print("initializeWebGPU")

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        # going to randomly generate an np buff to fill the screen

    def resizeWebGPU(self, width: int, height: int) -> None:
        """
        Resize the WebGPU context.

        This method handles resizing of the WebGPU context for the scene.

        Args:
            width (int): The new width of the widget.
            height (int): The new height of the widget.
        """
        print(f"resizeWebGPU {width} {height}")

    def timerEvent(self, event):
        self.update()


app = QApplication(sys.argv)
win = WebGPUScene()
win.show()
sys.exit(app.exec())
