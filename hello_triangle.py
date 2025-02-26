#!/usr/bin/env python3
import numpy as np
import wgpu
import wgpu.utils
from wgpu.gui.auto import WgpuCanvas
from wgpu.utils import get_default_device

# Define shader code
vertex_shader_code = """
struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) uv: vec2<f32>
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) fragColor: vec3<f32>,
    @location(1) fragUV: vec2<f32>
};

@vertex
fn main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    output.position = vec4<f32>(input.position, 1.0);
    output.fragColor = input.color;
    output.fragUV = input.uv;
    return output;
}
"""

fragment_shader_code = """
@fragment
fn main(@location(0) fragColor: vec3<f32>, @location(1) fragUV: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(fragColor, 1.0); // Simple color output
}
"""

# Triangle vertices (position, color, uv)
vertices = np.array(
    [
        # Position          # Color         # UV
        [0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0],  # Top vertex
        [-0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Bottom-left vertex
        [0.5, -0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # Bottom-right vertex
    ],
    dtype=np.float32,
)

# Get the default device
device = get_default_device()

# Create the vertex buffer
vertex_buffer = device.create_buffer_with_data(
    data=vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX
)

# Define the render pipeline
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])
render_pipeline = device.create_render_pipeline(
    layout=pipeline_layout,
    vertex={
        "module": device.create_shader_module(code=vertex_shader_code),
        "entry_point": "main",
        "buffers": [
            {
                "array_stride": 8 * 4,  # 8 floats per vertex
                "step_mode": "vertex",
                "attributes": [
                    {
                        "format": "float32x3",
                        "offset": 0,
                        "shader_location": 0,
                    },  # Position
                    {
                        "format": "float32x3",
                        "offset": 12,
                        "shader_location": 1,
                    },  # Color
                    {"format": "float32x2", "offset": 24, "shader_location": 2},  # UV
                ],
            }
        ],
    },
    fragment={
        "module": device.create_shader_module(code=fragment_shader_code),
        "entry_point": "main",
        "targets": [{"format": wgpu.TextureFormat.bgra8unorm}],
    },
    primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
)

# Create a canvas for rendering
canvas = WgpuCanvas(size=(640, 480), title="First Triangle")

# Render the triangle
command_encoder = device.create_command_encoder()
texture_view = canvas.get_current_texture_view()
render_pass = command_encoder.begin_render_pass(
    color_attachments=[
        {
            "view": texture_view,
            "resolve_target": None,
            "load_value": (0.1, 0.1, 0.1, 1),  # Background color
            "store_op": "store",
        }
    ]
)
render_pass.set_pipeline(render_pipeline)
render_pass.set_vertex_buffer(0, vertex_buffer)
render_pass.draw(3)  # Draw 3 vertices for the triangle
render_pass.end()
device.queue.submit([command_encoder.finish()])

canvas.present()
