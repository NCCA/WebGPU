from WebGPUWidget import WebGPUWidget
import nccapy
import numpy as np
import wgpu
import wgpu.backends.auto


class WebGPU(WebGPUWidget):
    def __init__(self):
        super().__init__()

    def initializeWebGPU(self):
        self.init_context()
        self.load_shaders("vertex_shader.wgsl", "fragment_shader.wgsl")
        self.create_geo()
        self.create_uniform_buffers()
        self.create_pipeline()
        self.rotation = 0.0

    def init_context(self, power_preference="high-performance", limits=None):
        # Request an adapter and device
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        self.device = self.adapter.request_device_sync(required_limits=limits)
        # this is the target texture size
        self.colour_texture = self.device.create_texture(
            size=(1024, 720, 1),  # width, height, depth
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.colour_buffer_view = self.colour_texture.create_view()
        print(vars(self.colour_buffer_view))
        # Now create a depth buffer
        depth_texture = self.device.create_texture(
            size=(1024, 720, 1),  # width, height, depth
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.depth_buffer_view = depth_texture.create_view()

        # Create the canvas and configure the swap chain

    def load_shaders(self, vertex_shader, fragment_shader):
        with open(vertex_shader, "r") as f:
            vertex_shader_code = f.read()

        with open(fragment_shader, "r") as f:
            fragment_shader_code = f.read()
        self.vertex_shader = self.device.create_shader_module(code=vertex_shader_code)
        self.fragment_shader = self.device.create_shader_module(code=fragment_shader_code)

    def create_geo(self):
        # Cube vertex data
        # fmt: off
        vertices = np.array([
            # Positions  # Colors
            -1, -1, -1, 1, 0, 0,
            1, -1, -1, 0, 1, 0,
            1, 1, -1, 0, 0, 1,
            -1, 1, -1, 1, 1, 0,
            -1, -1, 1, 1, 0, 1,
            1, -1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 1,
            -1, 1, 1, 0, 0, 0,
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4,
            0, 1, 5, 5, 4, 0,
            2, 3, 7, 7, 6, 2,
            0, 3, 7, 7, 4, 0,
            1, 2, 6, 6, 5, 1,
        ], dtype=np.uint16)
        # fmt: on
        # Create the vertex buffer
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertices, usage=wgpu.BufferUsage.VERTEX
        )

        # Create the index buffer
        self.index_buffer = self.device.create_buffer_with_data(
            data=self.indices, usage=wgpu.BufferUsage.INDEX
        )

    def create_pipeline(self):
        # Create the bind group
        bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                # Add other bindings as needed
            ]
        )

        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniform_buffer,
                        "offset": 0,
                        "size": self.mvp_matrix.nbytes,
                    },
                }
            ],
        )

        # Create the pipeline layout
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        # Create the render pipeline
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": self.vertex_shader,
                "entry_point": "main",
                "buffers": [
                    {
                        "array_stride": 6 * 4,
                        "attributes": [
                            {"shader_location": 0, "offset": 0, "format": "float32x3"},
                            {"shader_location": 1, "offset": 3 * 4, "format": "float32x3"},
                        ],
                    }
                ],
            },
            fragment={
                "module": self.fragment_shader,
                "entry_point": "main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
        )

    def _update_colour_buffer(self):
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
        self.buffer = np.frombuffer(raw_data, dtype=np.uint8).reshape(
            (1024, 720, 4)
        )  # Height, Width, Channels

        # Unmap the buffer when done
        readback_buffer.unmap()

    def update_uniform_buffers(self):
        self.rotation += 1
        x = nccapy.Mat4.rotate_x(self.rotation)
        y = nccapy.Mat4.rotate_y(self.rotation)
        z = nccapy.Mat4.rotate_z(self.rotation)
        rotation = x @ y @ z
        self.mvp_matrix = (self.persp @ self.lookat @ rotation).get_numpy().astype(np.float32)

        self.device.queue.write_buffer(
            buffer=self.uniform_buffer, buffer_offset=0, data=self.mvp_matrix.tobytes()
        )

    def paintWebGPU(self):
        command_encoder = self.device.create_command_encoder()
        render_pass = command_encoder.begin_render_pass(
            label="render_pass",
            color_attachments=[
                {
                    "view": self.colour_buffer_view,
                    "resolve_target": None,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0, 0, 0, 1),
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
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.set_index_buffer(self.index_buffer, wgpu.IndexFormat.uint16)
        render_pass.draw_indexed(len(self.indices), 1, 0, 0, 0)
        render_pass.end()
        # Submit the commands
        self.device.queue.submit([command_encoder.finish()])
        self._update_colour_buffer()

    def create_uniform_buffers(self):
        self.persp = nccapy.perspective(45.0, 1.0, 0.1, 100.0)
        self.lookat = nccapy.look_at(
            nccapy.Vec3(0, 0, 5), nccapy.Vec3(0, 0, 0), nccapy.Vec3(0, 1, 0)
        )
        rotation = nccapy.Mat4.rotate_y(40)
        self.mvp_matrix = (self.persp @ self.lookat @ rotation).get_numpy().astype(np.float32)

        self.uniform_buffer = self.device.create_buffer_with_data(
            data=self.mvp_matrix.astype(np.float32),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="uniform_buffer MVP",
        )

    def resizeWebGPU(self, width, height):
        pass
