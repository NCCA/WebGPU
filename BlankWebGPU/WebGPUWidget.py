from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np
from qtpy.QtCore import QRect, Qt
from qtpy.QtGui import QColor, QFont, QImage, QPainter
from qtpy.QtWidgets import QWidget
import wgpu

_TEXTURE_FORMAT = wgpu.TextureFormat.rgba8unorm


class QWidgetABCMeta(type(QWidget), ABCMeta):
    """
    A metaclass that combines the functionality of ABCMeta and QWidget's metaclass.

    This allows the creation of abstract base classes that are also QWidgets.
    """

    pass


class WebGPUWidget(QWidget, metaclass=QWidgetABCMeta):
    """
    An abstract base class for WebGPU widgets.

    This class provides a template for creating WebGPU widgets with methods
    that must be implemented in subclasses. It is designed to be similar to the QOpenGLWidget class.

    Attributes:
        initialized (bool): A flag indicating whether the widget has been initialized, default is False and will allow initializeWebGPU to be called once.
        The default context will generate a colour and depth buffer for the widget which will be automatically re-sized on resize events to the size of the screen
        This is the simplest method to deal with things, but may not be the most efficient for all use cases.
    """

    def __init__(self, width=1024, height=720) -> None:
        """
        Initialize the AbstractWebGPUWidget.

        This constructor initializes the QWidget and sets the initialized flag to False.
        """
        super().__init__()
        self.resize(width, height)
        self.initialized = False
        self.text_buffer: List[Tuple[int, int, str, int, str, QColor]] = []
        self.buffer = None
        self.texture_size = (self.width(), self.height(), 1)

    @abstractmethod
    def initializeWebGPU(self, default_context=True) -> None:
        """
        Initialize the WebGPU context.
        This method must be implemented in subclasses to set up the WebGPU context. Will be called once.
        We have an optional parameter to allow for the default context to be used or not.
        Parameters
        ----------
        default_context : bool
            A flag to indicate whether the default context should be used, default
            is True.
        """
        if default_context:
            self._init_default_context()

    def _init_default_context(self) -> None:
        """
        Initialize the default WebGPU context.

        This method initializes the default WebGPU context.
        """
        # Request an adapter and device
        self.adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        self.device = self.adapter.request_device_sync(required_limits=None)
        self._create_render_buffers()

    def _create_render_buffers(self):
        # this is the target texture size
        self.colour_texture = self.device.create_texture(
            size=self.texture_size,  # width, height, depth
            format=_TEXTURE_FORMAT,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.colour_buffer_view = self.colour_texture.create_view()
        # Now create a depth buffer
        depth_texture = self.device.create_texture(
            size=self.texture_size,  # width, height, depth
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.depth_buffer_view = depth_texture.create_view()

    @abstractmethod
    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method must be implemented in subclasses to render the WebGPU content. This will be called on every paint event
        and is where all the main rendering code should be placed.
        """
        pass

    @abstractmethod
    def resizeWebGPU(self, width: int, height: int) -> None:
        """
        Resize the WebGPU context.

        This method must be implemented in subclasses to handle resizing of the WebGPU context. Will be called on a resize of the widget.

        Args:
            width (int): The new width of the widget.
            height (int): The new height of the widget.
        """
        # set new texture size
        self.texture_size = (self.width(), self.height(), 1)
        # clean up old buffers
        self.colour_texture.destroy()
        self.depth_texture.destroy()
        # create new buffers
        self._create_render_buffers()

        pass

    def paintEvent(self, event) -> None:
        """
        Handle the paint event to render the WebGPU content.

        Args:
            event (QPaintEvent): The paint event.
        """
        if not self.initialized:
            self.initializeWebGPU()
            self.initialized = True
        self.paintWebGPU()
        painter = QPainter(self)

        if self.buffer is not None:
            self._present_image(painter, self.buffer)
        for x, y, text, size, font, colour in self.text_buffer:
            painter.setPen(colour)
            painter.setFont(QFont("Arial", size))
            painter.drawText(x, y, text)
        self.text_buffer.clear()

        return super().paintEvent(event)

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

    def resizeEvent(self, event) -> None:
        """
        Handle the resize event to adjust the WebGPU context.

        Args:
            event (QResizeEvent): The resize event.
        """
        self.resizeWebGPU(event.size().width(), event.size().height())
        return super().resizeEvent(event)

    def _present_image(self, painter, image_data: np.ndarray) -> None:
        """
        Present the image data on the canvas.

        Args:
            image_data (np.ndarray): The image data to render.
        """
        size = image_data.shape[0], image_data.shape[1]  # width, height
        # We want to simply blit the image (copy pixels one-to-one on framebuffer).
        # Just in case, set render hints that may hurt performance.
        painter.setRenderHints(
            painter.RenderHint.Antialiasing | painter.RenderHint.SmoothPixmapTransform,
            False,
        )

        image = QImage(
            image_data.flatten(),
            size[0],
            size[1],
            size[0] * 4,
            QImage.Format.Format_RGBA8888,
        )

        rect1 = QRect(0, 0, size[0], size[1])
        rect2 = self.rect()
        painter.drawImage(rect2, image, rect1)
