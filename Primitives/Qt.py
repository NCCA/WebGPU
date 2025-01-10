#!/usr/bin/env python3

import sys

import nccapy
from qtpy.QtCore import QRect, Qt
from qtpy.QtGui import QImage, QPainter
from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from WebGPU import WebGPU


class DrawingWidget(QWidget):
    def __init__(self, parent=None, size=(1024, 720)):
        super().__init__(parent)
        # self.setFixedSize(size[0], size[1])  # Set the size of the drawing widget

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
        self.setWindowTitle("WebGPU in Qt")
        self.webgpu = WebGPU()
        self.spinXFace = int(0)
        self.spinYFace = int(0)
        self.rotate = False
        self.translate = False
        self.origX = int(0)
        self.origY = int(0)
        self.origXPos = int(0)
        self.origYPos = int(0)
        self.INCREMENT = 0.01
        self.ZOOM = 0.1
        self.modelPos = nccapy.Vec3()

        # Create a central widget with the drawing widget
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.drawing_widget = DrawingWidget()
        layout.addWidget(self.drawing_widget)
        self.setCentralWidget(central_widget)
        self.resize(1024, 720)
        self.timer = self.startTimer(20)

    def timerEvent(self, event):
        self.webgpu.set_mouse(self.spinXFace, self.spinYFace, self.modelPos)
        self.webgpu.update_uniform_buffers()
        self.webgpu.render()
        self.drawing_widget.buffer = self.webgpu.get_colour_buffer()
        self.update()

    def mousePressEvent(self, event):
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            self.origX = pos.x()
            self.origY = pos.y()
            self.rotate = True

        elif event.button() == Qt.MouseButton.RightButton:
            self.origXPos = pos.x()
            self.origYPos = pos.y()
            self.translate = True

    def mouseMoveEvent(self, event):
        if self.rotate and event.buttons() == Qt.MouseButton.LeftButton:
            pos = event.position()
            diffx = int(pos.x() - self.origX)
            diffy = int(pos.y() - self.origY)
            self.spinXFace += 0.5 * diffy
            self.spinYFace += 0.5 * diffx
            self.origX = pos.x()
            self.origY = pos.y()
            self.update()
        elif self.translate and event.buttons() == Qt.MouseButton.RightButton:
            pos = event.position()
            diffX = int(pos.x() - self.origXPos)
            diffY = int(pos.y() - self.origYPos)
            self.origXPos = pos.x()
            self.origYPos = pos.y()
            self.modelPos.x += self.INCREMENT * diffX
            self.modelPos.y -= self.INCREMENT * diffY
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.rotate = False

        elif event.button() == Qt.MouseButton.RightButton:
            self.translate = False

    def wheelEvent(self, event):
        numPixels = event.pixelDelta()
        if numPixels.x() > 0:
            self.modelPos.z += self.ZOOM

        elif numPixels.x() < 0:
            self.modelPos.z -= self.ZOOM
        if numPixels.y() > 0:
            self.modelPos.x += self.ZOOM

        elif numPixels.y() < 0:
            self.modelPos.x -= self.ZOOM

        self.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            exit()
        elif key == Qt.Key.Key_Space:
            self.spinXFace = 0
            self.spinYFace = 0
            self.modelPos.set(0, 0, 0)
        elif key == Qt.Key.Key_L:
            self.transformLight ^= True
        self.update()


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
