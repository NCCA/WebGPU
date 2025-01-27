
from typing import List, Optional, Tuple

import numpy as np
from qtpy.QtCore import QRect, Qt
from qtpy.QtGui import QColor, QFont, QImage, QPainter
from qtpy.QtWidgets import QWidget


class WebGPUCanvas(QWidget):
    """
    A QWidget subclass for rendering WebGPU content and text.

    Attributes:
        text_buffer (List[Tuple[int, int, str, int, str, QColor]]): A list to store text rendering information.
    """

    def __init__(
        self, parent: Optional[QWidget] = None, size: Tuple[int, int] = (1024, 720)
    ) -> None:
        """
        Initialize the WebGPUCanvas.

        Args:
            parent (Optional[QWidget], optional): The parent widget. Defaults to None.
            size (Tuple[int, int], optional): The size of the canvas. Defaults to (1024, 720).
        """
        super().__init__(parent)
        # self.setFixedSize(size[0], size[1])  # Set the size of the drawing widget
        self.text_buffer: List[Tuple[int, int, str, int, str, QColor]] = []

    
    def paintEvent(self, event: QPainter) -> None:
        """
        Handle the paint event to render the image and text.

        Args:
            event (QPaintEvent): The paint event.
        """
        if hasattr(self, "buffer"):
            painter = QPainter(self)
            self._present_image(painter,self.buffer)
        for x, y, text, size, font, colour in self.text_buffer:
            painter.setPen(colour)
            painter.setFont(QFont("Arial", size))
            painter.drawText(x, y, text)

    def render_text(
        self,
        x: int,
        y: int,
        text: str,
        size: int = 10,
        font: str = "Arial",
        colour: QColor = Qt.black,
    ) -> None:
        """
        Add text to the buffer to be rendered on the canvas.

        Args:
            x (int): The x-coordinate of the text.
            y (int): The y-coordinate of the text.
            text (str): The text to render.
            size (int, optional): The font size of the text. Defaults to 10.
            font (str, optional): The font family of the text. Defaults to "Arial".
            colour (QColor, optional): The colour of the text. Defaults to Qt.black.
        """
        self.text_buffer.append((x, y, text, size, font, colour))
        self.update()


    def _present_image(self,painter, image_data: np.ndarray) -> None:
        """
        Present the image data on the canvas.

        Args:
            image_data (np.ndarray): The image data to render.
        """
        size = image_data.shape[0], image_data.shape[1]  # width, height
        # We want to simply blit the image (copy pixels one-to-one on framebuffer).
        # Maybe Qt does this when the sizes match exactly (like they do here).
        # Converting to a QPixmap and painting that only makes it slower.

        # Just in case, set render hints that may hurt performance.
        painter.setRenderHints(
            painter.RenderHint.Antialiasing | painter.RenderHint.SmoothPixmapTransform, False
        )

        image = QImage(
            image_data.flatten(), size[0], size[1], size[0] * 4, QImage.Format.Format_RGBA8888
        )

        rect1 = QRect(0, 0, size[0], size[1])
        rect2 = self.rect()
        painter.drawImage(rect2, image, rect1)
