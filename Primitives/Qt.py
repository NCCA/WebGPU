#!/usr/bin/env python3

import sys
import time
import nccapy
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from WebGPU import WebGPU

from WebGPUCanvas import WebGPUCanvas


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
        self.start_time = time.perf_counter()

        # Create a central widget with the drawing widget
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        self.drawing_widget = WebGPUCanvas()
        layout.addWidget(self.drawing_widget)
        self.setCentralWidget(central_widget)
        self.resize(1024, 720)
        self.timer = self.startTimer(20)

    def resizeEvent(self, event):
        self.webgpu.resize(event.size().width(), event.size().height())

    def update(self) :

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
        self.drawing_widget.render_text(10, 20, "Light Position: " + str(self.webgpu.light_pos),size=20,colour=Qt.yellow)
        self.webgpu.move_camera(x, y)
        self.webgpu.update_uniform_buffers()
        self.webgpu.render()
        self.drawing_widget.buffer = self.webgpu.get_colour_buffer()
        super().update()
        #print((time.perf_counter() - self.start_time) * 1000)


    def timerEvent(self, event):
        self.start_time = time.perf_counter()
        self.update()

    def mousePressEvent(self, event):
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            self.origX = pos.x()
            self.origY = pos.y()
            self.rotate = True
            self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            self.origXPos = pos.x()
            self.origYPos = pos.y()
            self.translate = True
            self.update()


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
            self.webgpu.camera.eye.set(0, 2, 5)
       
        elif key == Qt.Key.Key_W:
            self.webgpu.light_pos.z += 1.0
        elif key == Qt.Key.Key_S:
            self.webgpu.light_pos.z -= 1.0
        elif key == Qt.Key.Key_A:
            self.webgpu.light_pos.x -= 1.0
        elif key == Qt.Key.Key_D:
            self.webgpu.light_pos.x += 1.0
        elif key == Qt.Key.Key_Q:
            self.webgpu.light_pos.y -= 1.0
        elif key == Qt.Key.Key_E:
            self.webgpu.light_pos.y += 1.0

        self.update()


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
