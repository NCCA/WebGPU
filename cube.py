#!/usr/bin/env python

import nccapy
import numpy as np
import wgpu
import wgpu.backends.auto
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.utils import get_default_device

# Vertex shader code
with open("vertex_shader.wgsl", "r") as f:
    vertex_shader_code = f.read()

with open("fragment_shader.wgsl", "r") as f:
    fragment_shader_code = f.read()


# fmt: off
# Cube vertex data
vertices = np.array([ 
    # Positions       # Colors
    -1, -1, -1,       1, 0, 0,
     1, -1, -1,       0, 1, 0,
     1,  1, -1,       0, 0, 1,
    -1,  1, -1,       1, 1, 0,
    -1, -1,  1,       1, 0, 1,
     1, -1,  1,       0, 1, 1,
     1,  1,  1,       1, 1, 1,
    -1,  1,  1,       0, 0, 0,
], dtype=np.float32)

indices = np.array([
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4,
    0, 1, 5, 5, 4, 0,
    2, 3, 7, 7, 6, 2,
    0, 3, 7, 7, 4, 0,
    1, 2, 6, 6, 5, 1,
], dtype=np.uint16)
# fmt: on
# Create the device and queue
device = get_default_device()
# Create a canvas for rendering
# Request an adapter and device
power_preference = "high-performance"
limits = None
adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
device = adapter.request_device_sync(required_limits=limits)

# Create the canvas and configure the swap chain
canvas = WgpuCanvas()
context = canvas.get_context()
swap_chain_format = context.get_preferred_format(adapter)
context.configure(
    device=device, format=swap_chain_format, usage=wgpu.TextureUsage.RENDER_ATTACHMENT
)

queue = device.queue
size = canvas.get_context().get_current_texture().size
print(f"size: {size}")
# Create the shaders
vertex_shader = device.create_shader_module(code=vertex_shader_code)
fragment_shader = device.create_shader_module(code=fragment_shader_code)

# Create the vertex buffer
vertex_buffer = device.create_buffer_with_data(
    data=vertices, usage=wgpu.BufferUsage.VERTEX
)

# Create the index buffer
index_buffer = device.create_buffer_with_data(
    data=indices, usage=wgpu.BufferUsage.INDEX
)

# Create the uniform buffer

persp = nccapy.perspective(45.0, 1.0, 0.1, 100.0)
lookat = nccapy.look_at(
    nccapy.Vec3(0, 0, 5), nccapy.Vec3(0, 0, 0), nccapy.Vec3(0, 1, 0)
)
rotation = nccapy.Mat4.rotate_y(40)
mvp_matrix = mvp_matrixncca = (persp @ lookat @ rotation).get_numpy().astype(np.float32)
# print(mvp_matrix)
# print(mvp_matrixncca)


uniform_buffer = device.create_buffer_with_data(
    data=mvp_matrix.astype(np.float32),
    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    label="uniform_buffer MVP",
)


# Create the depth texture
depth_texture = device.create_texture(
    size=size,
    usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    format=wgpu.TextureFormat.depth24plus,
)
depth_texture_view = depth_texture.create_view()


# Create the bind group
bind_group_layout = device.create_bind_group_layout(
    entries=[
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.VERTEX,
            "buffer": {"type": wgpu.BufferBindingType.uniform},
        },
        # Add other bindings as needed
    ]
)


bind_group = device.create_bind_group(
    layout=bind_group_layout,
    entries=[
        {
            "binding": 0,
            "resource": {
                "buffer": uniform_buffer,
                "offset": 0,
                "size": mvp_matrix.nbytes,
            },
        }
    ],
)


# Create the pipeline layout
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])


# Create the render pipeline
pipeline = device.create_render_pipeline(
    layout=pipeline_layout,
    vertex={
        "module": vertex_shader,
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
        "module": fragment_shader,
        "entry_point": "main",
        "targets": [{"format": wgpu.TextureFormat.bgra8unorm_srgb}],
    },
    primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
    depth_stencil={
        "format": wgpu.TextureFormat.depth24plus,
        "depth_write_enabled": True,
        "depth_compare": wgpu.CompareFunction.less,
    },
    multisample={"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False},
)


command_encoder = device.create_command_encoder()

# texture_view = canvas.get_swap_chain_texture()
texture_view = canvas.get_context("wgpu").get_current_texture().create_view()
# depth_texture_view=canvas.get_context("wgpu").get_current_depth_texture().create_view()
print(vars(texture_view))
command_encoder = device.create_command_encoder()
render_pass = command_encoder.begin_render_pass(
    label="render_pass",
    color_attachments=[
        {
            "view": texture_view,
            "resolve_target": None,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0, 0, 0, 1),
        }
    ],
    depth_stencil_attachment={
        "view": depth_texture_view,
        "depth_load_op": wgpu.LoadOp.clear,
        "depth_store_op": wgpu.StoreOp.store,
        "depth_clear_value": 1.0,
    },
)

render_pass.set_pipeline(pipeline)
render_pass.set_bind_group(0, bind_group, [], 0, 999999)
render_pass.set_vertex_buffer(0, vertex_buffer)
render_pass.set_index_buffer(index_buffer, wgpu.IndexFormat.uint16)


render_pass.draw_indexed(len(indices), 1, 0, 0, 0)
render_pass.end()

# Submit the commands
queue.submit([command_encoder.finish()])


run()
