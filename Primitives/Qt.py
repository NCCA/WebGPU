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
        self.ZOOM = 0.5
        self.modelPos = nccapy.Vec3()
        self.key_pressed = set()

        # Create a central widget with the drawing widget
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.drawing_widget = DrawingWidget()
        layout.addWidget(self.drawing_widget)
        self.setCentralWidget(central_widget)
        self.resize(1024, 720)
        self.timer = self.startTimer(20)

    def resizeEvent(self, event):
        self.webgpu.resize(event.size().width(), event.size().height())

    def timerEvent(self, event):
        x = 0.0
        y = 0.0

        for k in self.key_pressed:
            if k == Qt.Key.Key_Left:
                y -= 0.1
            elif k == Qt.Key.Key_Right:
                y += 0.1
            elif k == Qt.Key.Key_Up:
                x += 0.1
            elif k == Qt.Key.Key_Down:
                x -= 0.1
        self.webgpu.move_camera(x, y)
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
        if event.buttons() == Qt.MouseButton.LeftButton:
            pos = event.position()
            diffx = int(pos.x() - self.origX)
            diffy = int(pos.y() - self.origY)
            self.spinXFace += 0.5 * diffy
            self.spinYFace += 0.5 * diffx
            self.origX = pos.x()
            self.origY = pos.y()
            self.webgpu.update_camera_vectors(diffx, diffy)
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
        # if numPixels.y() > 0:
        #     self.modelPos.x += self.ZOOM

        # elif numPixels.y() < 0:
        #     self.modelPos.x -= self.ZOOM

        self.update()

    def keyReleaseEvent(self, event):
        key = event.key()
        self.key_pressed.remove(key)

    def keyPressEvent(self, event):
        key = event.key()
        self.key_pressed.add(key)
        if key == Qt.Key.Key_Escape:
            exit()
        elif key == Qt.Key.Key_Space:
            self.spinXFace = 0
            self.spinYFace = 0
            self.modelPos.set(0, 0, 0)
        elif key == Qt.Key.Key_L:
            self.transformLight ^= True
        elif key == Qt.Key.Key_1:
            self.webgpu.prim_index -= 1
            if self.webgpu.prim_index < 0:
                self.webgpu.prim_index = 0

        elif key == Qt.Key.Key_2:
            self.webgpu.prim_index += 1
            if self.webgpu.prim_index >= 11:
                self.webgpu.prim_index = 11

        self.update()


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
