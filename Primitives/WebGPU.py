from WebGPUWidget import WebGPUWidget
import nccapy
import numpy as np
import wgpu
import wgpu.backends.auto
from qtpy.QtCore import Qt

from Primitives import Primitives
from Pipelines import Pipelines, _TEXTURE_FORMAT
import FirstPersonCamera
from typing import Optional, Dict, Tuple


class WebGPU(WebGPUWidget):
    """
    A class for creating a WebGPU widget.

    This class extends WebGPUWidget and provides functionality for rendering
    WebGPU content with a first-person camera and lighting.

    Attributes:
        texture_size (Tuple[int, int, int]): The size of the texture.
        width (int): The width of the widget.
        height (int): The height of the widget.
        rotation (int): The rotation angle of the prims update each frame .
        light_pos (nccapy.Vec3): The position of the light source.
        camera (FirstPersonCamera.FirstPersonCamera): The first-person camera.
    """

    def __init__(self, texture_size=(1024, 1024, 1)):
        """
        Initialize the WebGPU widget.

        Args:
            texture_size (Tuple[int, int, int], optional): The size of the texture. Defaults to (1024, 1024, 1).
        """
        super().__init__()
        self.texture_size = texture_size
        self.width = 1024
        self.height = 720
        self.rotation = 0
        self.light_pos = nccapy.Vec3(0.0, 2.0, 0.0)
        self.camera = FirstPersonCamera.FirstPersonCamera(
            nccapy.Vec3(0.0, 2.0, 5.0),
            nccapy.Vec3(0.0, 0.0, 0.0),
            nccapy.Vec3(0.0, 1.0, 0.0),
            45.0,
        )
        self.camera.set_projection(45.0, self.width / self.height, 0.1, 250.0)

    def initializeWebGPU(self):
        """
        Initialize the WebGPU context and do some setup.
        this is called only once.
        """
        self._init_context()
        Primitives.create_line_grid("grid", self.device, 5.5, 5.5, 12)
        Primitives.create_sphere("sphere", self.device, 1.0, 200)
        Primitives.load_default_primitives(self.device)
        Primitives.create_cone("cone", self.device, 0.5, 10, 20, 50)

        self.line_pipeline = Pipelines.create_line_pipeline("line", self.device)
        self.diffuse_tri_pipeline = Pipelines.create_diffuse_triangle_pipeline(
            "diffuse_tri", self.device
        )

        self._init_buffers()

    def _init_buffers(self) -> None:
        """
        setup the buffers for the pipelines.
        """
        self.line_pipeline.uniform_data["MVP"] = (
            self.camera.get_vp().get_numpy().flatten()
        )
        self.line_pipeline.uniform_data["colour"] = np.array([1.0, 1.0, 1.0, 1.0])

        # setup the buffer for the diffuse tri pipeline
        self.diffuse_tri_pipeline.uniform_data[0]["MVP"] = (
            self.camera.get_vp().get_numpy().flatten()
        )
        self.diffuse_tri_pipeline.uniform_data[0]["model_view"] = (
            self.camera.get_vp().get_numpy().flatten()
        )
        self.diffuse_tri_pipeline.uniform_data[0]["normal_matrix"] = (
            nccapy.Mat4().get_numpy().flatten()
        )
        self.diffuse_tri_pipeline.uniform_data[0]["colour"] = np.array(
            [0.0, 0.0, 1.0, 1.0]
        )

        self.diffuse_tri_pipeline.uniform_data[1]["light_pos"] = np.array(
            [self.light_pos.x, self.light_pos.y, self.light_pos.z, 1.0]
        )
        self.diffuse_tri_pipeline.uniform_data[1]["light_diffuse"] = np.array(
            [1.0, 1.0, 1.0, 1.0]
        )

    def _init_context(
        self,
        power_preference: str = "high-performance",
        limits: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context with the specified power preference and limits.

        Args:
            power_preference (str, optional): The power preference for the WebGPU context. Defaults to "high-performance".
            limits (Optional[Dict[str, int]], optional): A dictionary of limits for the WebGPU context. Defaults to None.
        """
        # Request an adapter and device
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        self.device = self.adapter.request_device_sync(required_limits=limits)
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

    def update_uniform_buffers(self) -> None:
        """
        update the uniform buffers for the line pipeline.
        """
        mvp_matrix = (self.camera.get_vp()).get_numpy().astype(np.float32)
        self.line_pipeline.uniform_data["MVP"] = mvp_matrix.flatten()
        self.device.queue.write_buffer(
            buffer=self.line_pipeline.uniform_buffer,
            buffer_offset=0,
            data=self.line_pipeline.uniform_data.tobytes(),
        )

    def _set_prim_uniforms(
        self,
        index: int,
        colour: Tuple[float, float, float, float],
        position: nccapy.Vec3,
        rotation: nccapy.Vec3 = nccapy.Vec3(0, 0, 0),
        scale: nccapy.Vec3 = nccapy.Vec3(1, 1, 1),
    ) -> None:
        """
        Set the uniforms for a primitive.

        This method sets the uniform values for a specified primitive, including its colour,
        position, rotation, and scale.

        Args:
            index (int): The index of the primitive.
            colour (Tuple[float, float, float, float]): The colour of the primitive as an RGBA tuple.
            position (nccapy.Vec3): The position of the primitive.
            rotation (nccapy.Vec3, optional): The rotation of the primitive. Defaults to nccapy.Vec3(0, 0, 0).
            scale (nccapy.Vec3, optional): The scale of the primitive. Defaults to nccapy.Vec3(1, 1, 1).
        """
        tx = nccapy.Transform()
        tx.set_position(position)
        tx.set_rotation(rotation)
        tx.set_scale(scale)

        mv_matrix = (self.camera.view @ tx.get_matrix()).get_numpy().astype(np.float32)

        mvp_matrix = (
            (self.camera.get_vp() @ tx.get_matrix()).get_numpy().astype(np.float32)
        )

        self.diffuse_tri_pipeline.uniform_data[0]["MVP"] = mvp_matrix.flatten()
        self.diffuse_tri_pipeline.uniform_data[0]["model_view"] = mv_matrix.flatten()

        nm = self.camera.view @ tx.get_matrix()
        # as we need only the rotation part of the model view matrix we can zero the rest
        nm.m[0][3] = 0
        nm.m[1][3] = 0
        nm.m[2][3] = 0
        nm.m[3][0] = 0
        nm.m[3][1] = 0
        nm.m[3][2] = 0
        nm.m[3][3] = 1
        nm = nm.inverse()
        nm.transpose()

        self.diffuse_tri_pipeline.uniform_data[0]["normal_matrix"] = (
            nm.get_numpy().flatten()
        )
        self.diffuse_tri_pipeline.uniform_data[0]["colour"] = np.array([colour])
        # copy sub data
        self.device.queue.write_buffer(
            buffer=self.diffuse_tri_pipeline.uniform_buffer[0],
            buffer_offset=index * 256,
            data=self.diffuse_tri_pipeline.uniform_data[0].tobytes(),
        )
        self.diffuse_tri_pipeline.uniform_data[1]["light_pos"] = np.array(
            [self.light_pos.x, self.light_pos.y, self.light_pos.z, 1.0]
        )
        self.diffuse_tri_pipeline.uniform_data[1]["light_diffuse"] = np.array(
            [10.0, 10.0, 10.0, 1.0]
        )

        self.device.queue.write_buffer(
            buffer=self.diffuse_tri_pipeline.uniform_buffer[1],
            buffer_offset=0,
            data=self.diffuse_tri_pipeline.uniform_data[1].tobytes(),
        )

    def update_camera_vectors(self, diffx: float, diffy: float) -> None:
        """
        Update the camera vectors based on mouse movement.

        This method processes the mouse movement to update the camera's direction vectors.

        Args:
            diffx (float): The difference in the x-coordinate of the mouse movement.
            diffy (float): The difference in the y-coordinate of the mouse movement.
        """
        self.camera.process_mouse_movement(diffx, diffy)

    def move_camera(self, x: float, y: float, delta: float = 0.1) -> None:
        """
        Move the camera based on input directions.

        This method moves the camera in the specified direction by a given delta.

        Args:
            x (float): The movement in the x-direction.
            y (float): The movement in the y-direction.
            delta (float, optional): The amount to move the camera. Defaults to 0.1.
        """
        self.camera.move(x, y, delta)

    def _update_colour_buffer(self) -> None:
        """
        Update the colour buffer.

        This method updates the colour buffer with the current colour data.
        """
        buffer_size = (
            1024 * 720 * 4
        )  # Width * Height * Bytes per pixel (RGBA8 is 4 bytes per pixel)
        readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )
        command_encoder = self.device.create_command_encoder()
        command_encoder.copy_texture_to_buffer(
            {"texture": self.colour_texture},
            {
                "buffer": readback_buffer,
                "bytes_per_row": 1024 * 4,  # Row stride (width * bytes per pixel)
                "rows_per_image": 720,  # Number of rows in the texture
            },
            (1024, 720, 1),  # Copy size: width, height, depth
        )
        self.device.queue.submit([command_encoder.finish()])
        # Map the buffer for reading
        readback_buffer.map_sync(mode=wgpu.MapMode.READ)

        # Access the mapped memory
        raw_data = readback_buffer.read_mapped()
        self.buffer = np.frombuffer(raw_data, dtype=np.uint8).reshape(
            (1024, 720, 4)
        )  # Height, Width, Channels

        # Unmap the buffer when done
        readback_buffer.unmap()

    def paintWebGPU(self):
        """
        called each time update is called render the frame here
        """
        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            label="render_pass",
            color_attachments=[
                {
                    "view": self.colour_buffer_view,
                    "resolve_target": None,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.7, 0.7, 0.7, 1),
                }
            ],
            depth_stencil_attachment={
                "view": self.depth_buffer_view,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        )
        # set grid uniforms
        self.update_uniform_buffers()

        render_pass.set_viewport(0, 0, 1024, 720, 0, 1)
        render_pass.set_pipeline(self.line_pipeline.pipeline)
        render_pass.set_bind_group(0, self.line_pipeline.bind_group, [], 0, 999999)
        Primitives.draw(render_pass, "grid")
        self.rotation += 1
        render_pass.set_pipeline(self.diffuse_tri_pipeline.pipeline)
        self._set_prim_uniforms(
            0,
            colour=[1.0, 0.0, 0.0, 1.0],
            position=[0.0, 0.5, 0.0],
            rotation=[self.rotation, 0, 0.0],
            scale=[1.0, 1.0, 1.0],
        )
        self._set_prim_uniforms(
            1,
            colour=[0.0, 1.0, 0.0, 1.0],
            position=[1.0, 0.5, 0.0],
            rotation=[0.0, self.rotation, 0.0],
            scale=[1.0, 1.0, 1.0],
        )
        self._set_prim_uniforms(
            2,
            colour=[0.0, 0.0, 1.0, 1.0],
            position=[-1.0, 0.5, 0.0],
            rotation=[0.0, 0, self.rotation],
            scale=[0.1, 0.1, 0.1],
        )
        # set lights
        render_pass.set_bind_group(1, self.diffuse_tri_pipeline.bind_group, [0])

        render_pass.set_bind_group(0, self.diffuse_tri_pipeline.bind_group, [0])
        Primitives.draw(render_pass, "troll")
        render_pass.set_bind_group(0, self.diffuse_tri_pipeline.bind_group, [1 * 256])
        Primitives.draw(render_pass, "teapot")

        render_pass.set_bind_group(0, self.diffuse_tri_pipeline.bind_group, [2 * 256])
        Primitives.draw(render_pass, "bunny")

        render_pass.end()

        # Submit the commands
        self.device.queue.submit([command_encoder.finish()])
        self.render_text(
            10, 20, "Light Position: " + str(self.light_pos), size=20, colour=Qt.yellow
        )

        self._update_colour_buffer()

    def resizeWebGPU(self, width: int, height: int) -> None:
        """
        Resize the WebGPU context.

        This method handles resizing of the WebGPU context when the widget is resized.

        Args:
            width (int): The new width of the widget.
            height (int): The new height of the widget.
        """
        try:
            aspect = width / height
        except ZeroDivisionError:
            aspect = 1
        self.width = width
        self.height = height
        self.camera.set_projection(self.camera.fov, aspect, 0.1, 250.0)
