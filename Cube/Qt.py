#!/usr/bin/env python3

import sys

from qtpy.QtCore import QRect
from qtpy.QtGui import QImage, QPainter
from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from WebGPU import WebGPU


class DrawingWidget(QWidget):
    def __init__(self, parent=None, size=(1024, 720)):
        super().__init__(parent)

    def paintEvent(self, event):
        if hasattr(self, "buffer"):
            self._present_image(self.buffer)

    def _present_image(self, image_data):
        size = image_data.shape[0], image_data.shape[1]  # width, height
        painter = QPainter(self)
        # # We want to simply blit the image (copy pixels one-to-one on framebuffer).
        # # Maybe Qt does this when the sizes match exactly (like they do here).
        # # Converting to a QPixmap and painting that only makes it slower.

        # # Just in case, set render hints that may hurt performance.
        painter.setRenderHints(
            painter.RenderHint.Antialiasing | painter.RenderHint.SmoothPixmapTransform, False
        )

        image = QImage(
            image_data.flatten(), size[0], size[1], size[0] * 4, QImage.Format.Format_RGBA8888
        )

        rect1 = QRect(0, 0, size[0], size[1])
        rect2 = self.rect()
        painter.drawImage(rect2, image, rect1)
        # Uncomment for testing purposes
        # painter.setPen(QColor("#0000ff"))
        # painter.setFont(QFont("Arial", 30))
        # painter.drawText(100, 100, "This is an image")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drawing Application")
        self.webgpu = WebGPU()

        # Create a central widget with the drawing widget
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.drawing_widget = DrawingWidget()
        layout.addWidget(self.drawing_widget)
        self.setCentralWidget(central_widget)
        self.resize(1024, 720)
        self.timer = self.startTimer(20)

    def timerEvent(self, event):
        self.webgpu.update_uniform_buffers()
        self.webgpu.render()
        self.drawing_widget.buffer = self.webgpu.get_colour_buffer()
        self.update()


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
