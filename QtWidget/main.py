#!/usr/bin/env python3
import sys

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from WebGPUWidget import WebGPUWidget


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
        self.startTimer(100)

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        self.render_text(10, 20, "PaintGPU Test", size=20, colour=Qt.yellow)

        # going to randomly generate an np buff to fill the screen
        self.buffer = np.random.randint(0, 255, (1024, 720, 4), dtype=np.uint8)

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
