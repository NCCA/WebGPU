import nccapy
import numpy as np
import wgpu

from Primitives import Primitives
from Pipelines import Pipelines


class WebGPU:
    def __init__(self, texture_size=(1024, 1024, 1)):
        self.rotation = 0.0
        self.mouse_rotation = nccapy.Mat4()
        self.texture_size = texture_size
        self.init_context()
        self.persp = nccapy.perspective(45.0, 1.0, 0.1, 100.0)
        self.lookat = nccapy.look_at(
            nccapy.Vec3(0, 0, 12), nccapy.Vec3(0, 0, 0), nccapy.Vec3(0, 1, 0)
        )

        Primitives.create_line_grid("grid", self.device, 5.5, 5.5, 12)
        Primitives.create_sphere("sphere", self.device, 1.0, 200)
        self.line_pipeline = Pipelines.create_line_pipeline("line", self.device)
        self.diffuse_pipeline=Pipelines.create_diffuse_pipeline("diffuse", self.device)
        self.init_buffers()

    def init_buffers(self):
        self.line_pipeline.uniform_data["MVP"] = nccapy.Mat4().get_numpy().flatten()
        self.line_pipeline.uniform_data["colour"] = np.array([1.0, 1.0, 0.0])

        # setup the buffer for the diffuse pipeline
        self.diffuse_pipeline.uniform_data[0]["MVP"] = nccapy.Mat4().get_numpy().flatten()
        self.diffuse_pipeline.uniform_data[0]["model_view"] = nccapy.Mat4().get_numpy().flatten()
        self.diffuse_pipeline.uniform_data[0]["normal_matrix"] = nccapy.Mat4().get_numpy().flatten()
        self.diffuse_pipeline.uniform_data[0]["colour"] = np.array([1.0, .0, 0.0])

        self.diffuse_pipeline.uniform_data[1]["light_pos"] = np.array([2.0, 2.0, 0.0])
        self.diffuse_pipeline.uniform_data[1]["light_diffuse"] = np.array([1.0, 1.0, 1.0])


    def init_context(self, power_preference="high-performance", limits=None):
        # Request an adapter and device
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        self.device = self.adapter.request_device_sync(required_limits=limits)
        # this is the target texture size
        self.colour_texture = self.device.create_texture(
            size=self.texture_size,  # width, height, depth
            format=wgpu.TextureFormat.rgba8unorm,
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
        self.rotation += 1
        # x = nccapy.Mat4.rotate_x(self.rotation)
        y = nccapy.Mat4.rotate_y(self.rotation)
        # z = nccapy.Mat4.rotate_z(self.rotation)
        rotation = y
        mvp_matrix = (
            (self.persp @ self.lookat @ self.mouse_rotation)
            .get_numpy()
            .astype(np.float32)
        )
        self.line_pipeline.uniform_data["MVP"] = mvp_matrix.flatten()

        self.device.queue.write_buffer(
            buffer=self.line_pipeline.uniform_buffer,
            buffer_offset=0,
            data=self.line_pipeline.uniform_data.tobytes(),
        )

    

        self.diffuse_pipeline.uniform_data[0]["MVP"] = mvp_matrix.flatten()

        mv_matrix = (
            (self.lookat @ self.mouse_rotation)
            .get_numpy()
            .astype(np.float32)
        )
        self.diffuse_pipeline.uniform_data[0]["model_view"]=mv_matrix.flatten()
        nm = (self.lookat @ self.mouse_rotation)
        nm.inverse()
        nm.transpose()
        self.diffuse_pipeline.uniform_data[0]["normal_matrix"]=nm.get_numpy().flatten()

        self.diffuse_pipeline.uniform_data[0]["colour"] =np.array([1.0, 0.0, 0.0])

        self.device.queue.write_buffer(
            buffer=self.diffuse_pipeline.uniform_buffer[0],
            buffer_offset=0,
            data=self.diffuse_pipeline.uniform_data[0].tobytes(),
        )

        self.device.queue.write_buffer(
            buffer=self.diffuse_pipeline.uniform_buffer[1],
            buffer_offset=0,
            data=self.diffuse_pipeline.uniform_data[1].tobytes(),
        )



    def set_mouse(self, x, y, model_pos):
        rot_x = nccapy.Mat4.rotate_x(x)
        rot_y = nccapy.Mat4.rotate_y(y)
        rotation = rot_x @ rot_y
        rotation.m[3][0] = model_pos.x
        rotation.m[3][1] = model_pos.y
        rotation.m[3][2] = model_pos.z
        self.mouse_rotation = rotation

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


        render_pass.set_pipeline(self.diffuse_pipeline.pipeline)
        render_pass.set_bind_group(0, self.diffuse_pipeline.bind_group[0], [], 0, 999999)
        render_pass.set_bind_group(1, self.diffuse_pipeline.bind_group[1], [], 0, 999999)
        Primitives.draw(render_pass, "sphere")


        render_pass.end()





        # Submit the commands
        self.device.queue.submit([command_encoder.finish()])

    # def create_uniform_buffers(self):
