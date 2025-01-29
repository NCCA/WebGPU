#!/usr/bin/env python3

import sys
from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from WebGPU import WebGPU


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WebGPU in Qt")
        self.webgpu = WebGPU()
        # Create a central widget with the drawing widget
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.webgpu)
        self.setCentralWidget(central_widget)
        self.resize(1024, 720)
        self.timer = self.startTimer(20)

    def timerEvent(self, event):
        # update the cube rotation
        self.webgpu.update_uniform_buffers()
        self.update()


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
