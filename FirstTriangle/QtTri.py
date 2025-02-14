#!/usr/bin/env python3
import sys

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication
from WebGPUWidget import WebGPUWidget
import wgpu
import wgpu.utils
from wgpu.utils import get_default_device


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of AbstractWebGPUWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.

        Args:
            device (wgpu.GPUDevice): The GPU device.

        Returns:
            wgpu.GPURenderPipeline: The created render pipeline.
        """
        vertex_shader_code = """
        struct VertexIn {
            @location(0) position: vec3<f32>,
            @location(1) color: vec3<f32>,
        };

        struct VertexOut {
            @builtin(position) position: vec4<f32>,
            @location(0) fragColor: vec3<f32>,
        };

        @vertex
        fn main(input: VertexIn) -> VertexOut {
            var output: VertexOut;
            output.position = vec4<f32>(input.position, 1.0);
            output.fragColor = input.color;
            return output;
        }
        """

        fragment_shader_code = """
        @fragment
        fn main(@location(0) fragColor: vec3<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(fragColor, 1.0); // Simple color output
        }
        """

        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[])
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": self.device.create_shader_module(code=vertex_shader_code),
                "entry_point": "main",
                "buffers": [
                    {
                        "array_stride": 6 * 4,
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": self.device.create_shader_module(code=fragment_shader_code),
                "entry_point": "main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

    def __init__(self):
        super().__init__()
        self.device = None
        self.pipeline = None
        self.buffer = None
        self.angle = 0.0
        self.vertices = np.array(
            [
                [0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
                [-0.5, -0.5, 0.0, 0.0, 1.0, 0.0],
                [0.5, -0.5, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.width = 1024
        self.height = 1024

    def initializeWebGPU(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        print("initializeWebGPU")
        self.device = get_default_device()

        self.vertex_buffer = self.device.create_buffer_with_data(
            data=self.vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

        self._create_render_pipeline()
        self.startTimer(100)

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.

        This method renders the WebGPU content for the scene.
        """
        self.render_text(10, 20, "First Triangle", size=20, colour=Qt.black)
        texture = self.device.create_texture(
            size=(self.width, self.height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        texture_view = texture.create_view()

        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": texture_view,
                    "resolve_target": None,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (1.0, 1.0, 1.0, 1.0),
                }
            ]
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(3)
        render_pass.end()
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer(texture)

    def _update_colour_buffer(self, texture) -> None:
        buffer_size = (
            self.width * self.height * 4
        )  # Width * Height * Bytes per pixel (RGBA8 is 4 bytes per pixel)
        readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        command_encoder = self.device.create_command_encoder()
        command_encoder.copy_texture_to_buffer(
            {"texture": texture},
            {
                "buffer": readback_buffer,
                "bytes_per_row": self.width * 4,  # Row stride (width * bytes per pixel)
                "rows_per_image": self.height,  # Number of rows in the texture
            },
            (self.width, self.height, 1),  # Copy size: width, height, depth
        )
        self.device.queue.submit([command_encoder.finish()])
        # Map the buffer for reading
        readback_buffer.map_sync(mode=wgpu.MapMode.READ)

        # Access the mapped memory
        raw_data = readback_buffer.read_mapped()
        self.buffer = np.frombuffer(raw_data, dtype=np.uint8).reshape(
            ((self.width, self.height, 4))
        )  # Height, Width, Channels

        # Unmap the buffer when done
        readback_buffer.unmap()

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
        self.angle += 0.1
        vertices = self._rotate()
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

        self.update()

    def _rotate(self) -> np.ndarray:
        """
        Rotate the vertices.
        """
        rotation_matrix = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle), 0.0],
                [np.sin(self.angle), np.cos(self.angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        rotated_vertices = self.vertices.copy()

        rotated_vertices[:, :3] = np.dot(self.vertices[:, :3], rotation_matrix.T)
        return rotated_vertices


app = QApplication(sys.argv)
win = WebGPUScene()
win.show()
sys.exit(app.exec())
