#!/usr/bin/env python

import wgpu
import wgpu.backends.auto
import numpy as np
from pyrr import Matrix44, Vector3

# Vertex shader code
vertex_shader_code = """
struct Uniforms {
    mvpMatrix : mat4x4<f32>;
};

struct VertexInput {
    @location(0) position : vec3<f32>;
    @location(1) color : vec3<f32>;
};

struct VertexOutput {
    @builtin(position) position : vec4<f32>;
    @location(0) color : vec3<f32>;
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn main(input : VertexInput) -> VertexOutput {
    var output : VertexOutput;
    output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 1.0);
    output.color = input.color;
    return output;
}
"""

# Fragment shader code
fragment_shader_code = """
struct FragmentInput {
    @location(0) color : vec3<f32>;
};

struct FragmentOutput {
    @location(0) color : vec4<f32>;
};

@fragment
fn main(input : FragmentInput) -> FragmentOutput {
    var output : FragmentOutput;
    output.color = vec4<f32>(input.color, 1.0);
    return output;
}
"""

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

# Create the device and queue
#adapter = wgpu.request_adapter(canvas=None, power_preference="high-performance")
#device = adapter.request_device(extensions=[], limits={})
power_preference="high-performance"
limits=None
adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
device = adapter.request_device_sync(required_limits=limits)

queue = device.queue

# Create the shaders
vertex_shader = device.create_shader_module(code=vertex_shader_code)
fragment_shader = device.create_shader_module(code=fragment_shader_code)

# Create the pipeline layout
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

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
        "targets": [{"format": wgpu.TextureFormat.bgra8unorm}],
    },
    primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
)

# Create the vertex buffer
vertex_buffer = device.create_buffer_with_data(data=vertices, usage=wgpu.BufferUsage.VERTEX)

# Create the index buffer
index_buffer = device.create_buffer_with_data(data=indices, usage=wgpu.BufferUsage.INDEX)

# Create the uniform buffer
mvp_matrix = Matrix44.perspective_projection(45.0, 1.0, 0.1, 100.0) * Matrix44.look_at(
    eye=Vector3([3, 3, 3]),
    target=Vector3([0, 0, 0]),
    up=Vector3([0, 1, 0])
)
uniform_buffer = device.create_buffer_with_data(data=mvp_matrix.astype(np.float32), usage=wgpu.BufferUsage.UNIFORM)

# Create the bind group
bind_group_layout = device.create_bind_group_layout(entries=[
    {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX, "type": wgpu.BindingType.uniform_buffer},
])
bind_group = device.create_bind_group(layout=bind_group_layout, entries=[
    {"binding": 0, "resource": {"buffer": uniform_buffer, "offset": 0, "size": mvp_matrix.nbytes}},
])

# Create the render pass
texture = device.create_texture(
    size=(640, 480, 1),
    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    format=wgpu.TextureFormat.bgra8unorm,
)
texture_view = texture.create_view()
depth_texture = device.create_texture(
    size=(640, 480, 1),
    usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    format=wgpu.TextureFormat.depth24plus_stencil8,
)
depth_texture_view = depth_texture.create_view()

command_encoder = device.create_command_encoder()
render_pass = command_encoder.begin_render_pass(
    color_attachments=[{
        "attachment": texture_view,
        "resolve_target": None,
        "load_value": (0, 0, 0, 1),
        "store_op": wgpu.StoreOp.store,
    }],
    depth_stencil_attachment={
        "attachment": depth_texture_view,
        "depth_load_value": 1.0,
        "depth_store_op": wgpu.StoreOp.store,
        "stencil_load_value": 0,
        "stencil_store_op": wgpu.StoreOp.store,
    },
)

render_pass.set_pipeline(pipeline)
render_pass.set_bind_group(0, bind_group, [], 0, 999999)
render_pass.set_vertex_buffer(0, vertex_buffer)
render_pass.set_index_buffer(index_buffer, wgpu.IndexFormat.uint16)
render_pass.draw_indexed(len(indices), 1, 0, 0, 0)
render_pass.end_pass()

# Submit the commands
queue.submit([command_encoder.finish()])

# Save the result to an image file
import imageio
image = np.zeros((480, 640, 4), dtype=np.uint8)
device.queue.read_texture(
    {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
    {"buffer": image, "offset": 0, "bytes_per_row": 640 * 4, "rows_per_image": 480},
    (640, 480, 1),
)
imageio.imwrite("output.png", image)