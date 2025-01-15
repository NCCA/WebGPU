import nccapy
import numpy as np
import wgpu
import math
from Primitives import Primitives
from Pipelines import Pipelines, _TEXTURE_FORMAT
from FirstPersonCamera import FirstPersonCamera


class WebGPU:
    def __init__(self, texture_size=(1024, 1024, 1)):
        self.prim_index = 0
        self.texture_size = texture_size
        self.init_context()
        self.width = 1024
        self.height = 720
        self.rotation=0
        Primitives.create_line_grid("grid", self.device, 5.5, 5.5, 12)
        Primitives.create_sphere("sphere", self.device, 1.0, 200)
        Primitives.load_default_primitives(self.device)
        Primitives.create_cone("cone", self.device, 0.5, 10, 20, 50)
        self.camera = FirstPersonCamera(
            nccapy.Vec3(0, 2, 5), nccapy.Vec3(0, 0, 0), nccapy.Vec3(0, 1, 0), 45.0
        )
        self.line_pipeline = Pipelines.create_line_pipeline("line", self.device)
        self.diffuse_tri_pipeline = Pipelines.create_diffuse_triangle_pipeline(
            "diffuse_tri", self.device
        )
        self.init_buffers()

    def init_buffers(self):
        self.line_pipeline.uniform_data["MVP"] = self.camera.get_vp().get_numpy().flatten()
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
        self.diffuse_tri_pipeline.uniform_data[0]["colour"] = np.array([0.0, 0.0, 1.0, 1.0])

        self.diffuse_tri_pipeline.uniform_data[1]["light_pos"] = np.array([0.0, 2.0, 0.0, 1.0])
        self.diffuse_tri_pipeline.uniform_data[1]["light_diffuse"] = np.array([1.0, 1.0, 1.0, 1.0])

    def init_context(self, power_preference="high-performance", limits=None):
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

    def get_colour_buffer(self):
        buffer_size = (
            1024 * 720 * 4
        )  # Width * Height * Bytes per pixel (RGBA8 is 4 bytes per pixel)
        readback_buffer = self.device.create_buffer(
            size=buffer_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
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
        pixel_data = np.frombuffer(raw_data, dtype=np.uint8).reshape(
            (1024, 720, 4)
        )  # Height, Width, Channels

        # Unmap the buffer when done
        readback_buffer.unmap()
        return pixel_data

    def update_uniform_buffers(self):
        mvp_matrix = (self.camera.get_vp()).get_numpy().astype(np.float32)
        self.line_pipeline.uniform_data["MVP"] = mvp_matrix.flatten()

        self.device.queue.write_buffer(
            buffer=self.line_pipeline.uniform_buffer,
            buffer_offset=0,
            data=self.line_pipeline.uniform_data.tobytes(),
        )

    def set_prim_uniforms(
        self, index, colour, position, rotation=nccapy.Vec3(0, 0, 0), scale=nccapy.Vec3(1, 1, 1)
    ):
        tx = nccapy.Transform()
        tx.set_position(position)
        tx.set_rotation(rotation)
        tx.set_scale(scale)

        mv_matrix = (self.camera.view @ tx.get_matrix()).get_numpy().astype(np.float32)

        mvp_matrix = (self.camera.get_vp() @ tx.get_matrix()).get_numpy().astype(np.float32)

        self.diffuse_tri_pipeline.uniform_data[0]["MVP"] = mvp_matrix.flatten()
        self.diffuse_tri_pipeline.uniform_data[0]["model_view"] = mv_matrix.flatten()
        nm = self.camera.get_vp() @ tx.get_matrix()
        nm.inverse()
        nm.transpose()
        self.diffuse_tri_pipeline.uniform_data[0]["normal_matrix"] = nm.get_numpy().flatten()

        self.diffuse_tri_pipeline.uniform_data[0]["colour"] = np.array([colour])
        # copy sub data
        self.device.queue.write_buffer(
            buffer=self.diffuse_tri_pipeline.uniform_buffer[0],
            buffer_offset=index * 256,
            data=self.diffuse_tri_pipeline.uniform_data[0].tobytes(),
        )
        self.diffuse_tri_pipeline.uniform_data[1]["light_pos"] = np.array([0.0, 5.0, -1.0, 1.0])
        self.diffuse_tri_pipeline.uniform_data[1]["light_diffuse"] = np.array([1.0, 1.0, 1.0, 1.0])

        self.device.queue.write_buffer(
            buffer=self.diffuse_tri_pipeline.uniform_buffer[1],
            buffer_offset=0,
            data=self.diffuse_tri_pipeline.uniform_data[1].tobytes(),
        )

    def update_camera_vectors(self, diffx, diffy):
        self.camera.process_mouse_movement(diffx, diffy)

    def move_camera(self, x, y, delta=0.1):
        self.camera.move(x, y, delta)

    def resize(self, width, height):
        # self.texture_size = (width, height, 1)
        # self.init_context()
        print("resize")
        try:
            aspect = width / height
        except ZeroDivisionError:
            aspect = 1
        self.camera.set_projection(45.0, aspect, 0.1, 250.0)
        self.width = width
        self.height = height

    def render(self):
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
        render_pass.set_viewport(0, 0, 1024, 720, 0, 1)
        render_pass.set_pipeline(self.line_pipeline.pipeline)
        render_pass.set_bind_group(0, self.line_pipeline.bind_group, [], 0, 999999)
        Primitives.draw(render_pass, "grid")
        self.rotation +=1
        render_pass.set_pipeline(self.diffuse_tri_pipeline.pipeline)
        self.set_prim_uniforms(
            0,
            colour=[1.0, 0.0, 0.0, 1.0],
            position=[0.0, 0.5, 0.0],
            rotation=[self.rotation, 0, 0.0],
            scale=[1.0, 1.0, 1.0],
        )
        self.set_prim_uniforms(
            1,
            colour=[0.0, 1.0, 0.0, 1.0],
            position=[1.0, 0.5, 0.0],
            rotation=[0.0, self.rotation, 0.0],
            scale=[1.0, 1.0, 1.0],
        )
        self.set_prim_uniforms(
            2,
            colour=[0.0, 0.0, 1.0, 1.0],
            position=[-1.0, 0.5, 0.0],
            rotation=[0.0, 0, self.rotation],
            scale=[0.1, 0.1, 0.1],
        )
        # set lights
        render_pass.set_bind_group(1, self.diffuse_tri_pipeline.bind_group[1], [])
        # set everything else
        # render_pass.set_bind_group(0, self.diffuse_tri_pipeline.bind_group[0], [0])
        # Primitives.draw(render_pass, "troll")
                # Ensure the offset does not exceed the buffer size
        
        render_pass.set_bind_group(0, self.diffuse_tri_pipeline.bind_group[0], [0])
        Primitives.draw(render_pass, "troll")
        render_pass.set_bind_group(0, self.diffuse_tri_pipeline.bind_group[0], [1*256],)
        Primitives.draw(render_pass, "teapot")

        render_pass.set_bind_group(0, self.diffuse_tri_pipeline.bind_group[0], [2*256])
        Primitives.draw(render_pass, "bunny")

        # render_pass.set_bind_group(0, self.diffuse_tri_pipeline.bind_group[0], [2*256],0,256)
        # Primitives.draw(render_pass, "troll")

        render_pass.end()

        # Submit the commands
        self.device.queue.submit([command_encoder.finish()])

