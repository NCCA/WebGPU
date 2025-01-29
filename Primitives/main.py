#!/usr/bin/env python3

import sys
import nccapy
import time
from qtpy.QtCore import Qt, QEvent
from qtpy.QtGui import QMouseEvent, QWheelEvent
from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from WebGPU import WebGPU
from typing import Set


class MainWindow(QMainWindow):
    """
    The main window for the WebGPU application.

    This class creates a main window with a WebGPU widget and handles user input
    for camera movement and interaction.

    Attributes:
        spinXFace (int): The rotation angle around the X-axis.
        spinYFace (int): The rotation angle around the Y-axis.
        rotate (bool): A flag indicating whether the widget is being rotated.
        translate (bool): A flag indicating whether the widget is being translated.
        origX (int): The original X-coordinate of the mouse press.
        origY (int): The original Y-coordinate of the mouse press.
        INCREMENT (float): The increment value for rotation.
        ZOOM (float): The zoom increment value.
        modelPos (nccapy.Vec3): The position of the model.
        key_pressed (Set[int]): A set of currently pressed keys.
        start_time (float): The start time of the application.
        webgpu (WebGPU): The WebGPU widget.
    """

    def __init__(self) -> None:
        """
        Initialize the main window.
        """
        super().__init__()
        self.setWindowTitle("WebGPU in Qt")
        self.spinXFace: int = 0
        self.spinYFace: int = 0
        self.rotate: bool = False
        self.translate: bool = False
        self.origX: int = 0
        self.origY: int = 0
        self.INCREMENT: float = 0.01
        self.ZOOM: float = 0.5
        self.modelPos: nccapy.Vec3 = nccapy.Vec3()
        self.key_pressed: Set[int] = set()
        self.start_time: float = time.perf_counter()

        # Create a central widget with the WebGPU widget
        self.webgpu: WebGPU = WebGPU()
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.webgpu)
        self.setCentralWidget(central_widget)
        self.resize(1024, 720)
        self.startTimer(10)

    def timerEvent(self, event):
        self.update()

    def resizeEvent(self, event: QEvent) -> None:
        """
        Handle the resize event to adjust the WebGPU widget.

        Args:
            event (QEvent): The resize event.
        """
        self.webgpu.resize(event.size().width(), event.size().height())

    def update(self) -> None:
        """
        Update the main window.

        This method updates the camera position based on the pressed keys.
        """
        x = 0.0
        y = 0.0
        UPDATE = 0.1
        for k in self.key_pressed:
            if k == Qt.Key.Key_Left:
                y -= UPDATE
            elif k == Qt.Key.Key_Right:
                y += UPDATE
            elif k == Qt.Key.Key_Up:
                x += UPDATE
            elif k == Qt.Key.Key_Down:
                x -= UPDATE
        self.webgpu.move_camera(x, y)
        super().update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Handle the mouse press event.

        Args:
            event (QMouseEvent): The mouse press event.
        """
        pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            self.origX = pos.x()
            self.origY = pos.y()
            self.rotate = True
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """
        Handle the mouse move event.

        Args:
            event (QMouseEvent): The mouse move event.
        """
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

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """
        Handle the mouse release event.

        Args:
            event (QMouseEvent): The mouse release event.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.rotate = False

        elif event.button() == Qt.MouseButton.RightButton:
            self.translate = False
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """
        Handle the mouse wheel event.

        Args:
            event (QWheelEvent): The mouse wheel event.
        """
        numPixels = event.pixelDelta()
        if numPixels.x() > 0:
            self.modelPos.z += self.ZOOM

        elif numPixels.x() < 0:
            self.modelPos.z -= self.ZOOM

        self.update()

    def keyReleaseEvent(self, event: QEvent) -> None:
        """
        Handle the key release event.

        Args:
            event (QEvent): The key release event.
        """
        key = event.key()
        self.key_pressed.remove(key)
        self.update()

    def keyPressEvent(self, event: QEvent) -> None:
        """
        Handle the key press event.

        Args:
            event (QEvent): The key press event.
        """
        key = event.key()
        self.key_pressed.add(key)
        if key == Qt.Key.Key_Escape:
            exit()
        elif key == Qt.Key.Key_Space:
            self.spinXFace = 0
            self.spinYFace = 0
            self.modelPos.set(0, 0, 0)
            self.webgpu.camera.eye.set(0.0, 2.0, 5.0)
            self.webgpu.light_pos.set(0.0, 2.0, 0.0)
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


def main() -> None:
    """
    The main entry point for the application.
    """
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
