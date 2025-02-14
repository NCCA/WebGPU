#!/usr/bin/env python3

import numpy as np
import wgpu
import wgpu.utils
from wgpu.utils import get_default_device
from PIL import Image


def create_vertex_buffer(
    device: wgpu.GPUDevice, vertices: np.ndarray
) -> wgpu.GPUBuffer:
    """
    Create a vertex buffer.

    Args:
        device (wgpu.GPUDevice): The GPU device.
        vertices (np.ndarray): The vertex data.

    Returns:
        wgpu.GPUBuffer: The created vertex buffer.
    """
    return device.create_buffer_with_data(
        data=vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX
    )


def create_render_pipeline(device: wgpu.GPUDevice) -> wgpu.GPURenderPipeline:
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

    pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])
    return device.create_render_pipeline(
        layout=pipeline_layout,
        vertex={
            "module": device.create_shader_module(code=vertex_shader_code),
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
            "module": device.create_shader_module(code=fragment_shader_code),
            "entry_point": "main",
            "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
        },
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
    )


def render_triangle(
    device: wgpu.GPUDevice,
    pipeline: wgpu.GPURenderPipeline,
    vertex_buffer: wgpu.GPUBuffer,
    width: int,
    height: int,
) -> wgpu.GPUTexture:
    """
    Render a triangle to a texture.

    Args:
        device (wgpu.GPUDevice): The GPU device.
        pipeline (wgpu.GPURenderPipeline): The render pipeline.
        vertex_buffer (wgpu.GPUBuffer): The vertex buffer.
        width (int): The width of the texture.
        height (int): The height of the texture.

    Returns:
        wgpu.GPUTexture: The rendered texture.
    """
    texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    texture_view = texture.create_view()

    command_encoder = device.create_command_encoder()
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
    render_pass.set_pipeline(pipeline)
    render_pass.set_vertex_buffer(0, vertex_buffer)
    render_pass.draw(3)
    render_pass.end()
    device.queue.submit([command_encoder.finish()])

    return texture


def copy_texture_to_buffer(
    device: wgpu.GPUDevice, texture: wgpu.GPUTexture, width: int, height: int
) -> np.ndarray:
    """
    Copy the texture to a buffer and return it as a NumPy array.

    Args:
        device (wgpu.GPUDevice): The GPU device.
        texture (wgpu.GPUTexture): The texture to copy.
        width (int): The width of the texture.
        height (int): The height of the texture.

    Returns:
        np.ndarray: The texture data as a NumPy array.
    """
    buffer_size = width * height * 4
    readback_buffer = device.create_buffer(
        size=buffer_size, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ
    )

    command_encoder = device.create_command_encoder()
    command_encoder.copy_texture_to_buffer(
        {"texture": texture},
        {
            "buffer": readback_buffer,
            "bytes_per_row": width * 4,
            "rows_per_image": height,
        },
        (width, height, 1),
    )
    device.queue.submit([command_encoder.finish()])

    readback_buffer.map_sync(mode=wgpu.MapMode.READ)
    raw_data = readback_buffer.read_mapped()
    buffer = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 4))
    readback_buffer.unmap()

    return buffer


def save_numpy_to_image(array: np.ndarray, filename="output.png"):
    """
    Save a NumPy array as an image using Pillow (PIL).

    Args:
        array (np.ndarray): The input array.
        filename (str): The filename to save the image as.
    """
    img = Image.fromarray(array)
    img.save(filename)


def main() -> None:
    """
    Main function to render a rotating triangle and save the image to a file.
    """
    vertices = np.array(
        [
            [0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
            [-0.5, -0.5, 0.0, 0.0, 1.0, 0.0],
            [0.5, -0.5, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    device = get_default_device()
    pipeline = create_render_pipeline(device)
    vertex_buffer = create_vertex_buffer(device, vertices)
    texture = render_triangle(device, pipeline, vertex_buffer, WIDTH, HEIGHT)
    buffer = copy_texture_to_buffer(device, texture, WIDTH, HEIGHT)
    save_numpy_to_image(buffer, "output.png")


if __name__ == "__main__":
    WIDTH = 1024
    HEIGHT = 720
    main()
