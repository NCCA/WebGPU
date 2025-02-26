#!/usr/bin/env -S uv run --script
import wgpu
import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.utils import get_default_device


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

# Vertex shader (WGSL)
shader_code = """
@vertex
fn vs_main(
    @location(0) quad_pos: vec2<f32>,  // Quad vertices
    @location(1) instance_pos: vec3<f32>,  // Instance positions
    @location(2) size: f32,  // Billboard size
    @builtin(instance_index) instanceID: u32
) -> @builtin(position) vec4<f32> {
    var cam_up: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);
    var cam_right: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0);

    let world_pos = instance_pos + cam_right * quad_pos.x * size + cam_up * quad_pos.y * size;
    return vec4<f32>(world_pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.2, 1.0);  // Orange color
}
"""

# Quad vertices (for billboards)
quad_vertices = np.array(
    [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]],
    dtype=np.float32,
)

# Instance positions (random points in space)
instance_data = np.array(
    [
        [0.0, 0.0, 0.0, 0.1],  # x, y, z, size
        [0.5, 0.5, 0.0, 0.2],
        [-0.5, -0.5, 0.0, 0.15],
    ],
    dtype=np.float32,
)

# Create vertex buffers
quad_buffer = device.create_buffer_with_data(
    data=quad_vertices, usage=wgpu.BufferUsage.VERTEX
)

instance_buffer = device.create_buffer_with_data(
    data=instance_data, usage=wgpu.BufferUsage.VERTEX
)

# Shader module
shader_module = device.create_shader_module(code=shader_code)

# Pipeline layout
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

# Vertex buffer layouts
vertex_buffers = [
    # Quad vertices layout (2D)
    {
        "array_stride": 2 * 4,  # vec2<f32>
        "step_mode": wgpu.VertexStepMode.vertex,
        "attributes": [
            {"format": wgpu.VertexFormat.float32x2, "offset": 0, "shader_location": 0}
        ],
    },
    # Instance attributes layout (vec3 pos + float size)
    {
        "array_stride": 4 * 4,  # vec3<f32> + float
        "step_mode": wgpu.VertexStepMode.instance,
        "attributes": [
            {"format": wgpu.VertexFormat.float32x3, "offset": 0, "shader_location": 1},
            {"format": wgpu.VertexFormat.float32, "offset": 12, "shader_location": 2},
        ],
    },
]

# Create render pipeline
pipeline = device.create_render_pipeline(
    layout=pipeline_layout,
    vertex={
        "module": shader_module,
        "entry_point": "vs_main",
        "buffers": vertex_buffers,
    },
    fragment={
        "module": shader_module,
        "entry_point": "fs_main",
        "targets": [{"format": wgpu.TextureFormat.bgra8unorm_srgb}],
    },
    primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
)


# Render function
def render():
    texture_view = canvas.get_context("wgpu").get_current_texture().create_view()

    command_encoder = device.create_command_encoder()
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": texture_view,
                "resolve_target": None,
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
                "clear_value": (0, 0, 0, 1),
            }
        ],
    )

    render_pass.set_pipeline(pipeline)
    render_pass.set_vertex_buffer(0, quad_buffer)
    render_pass.set_vertex_buffer(1, instance_buffer)
    render_pass.draw(6, len(instance_data))  # 6 vertices per quad, N instances
    render_pass.end()

    device.queue.submit([command_encoder.finish()])
    canvas.request_draw()


# Run the app
canvas.request_draw(render)
run()
