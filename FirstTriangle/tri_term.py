#!/usr/bin/env python3

import numpy as np
import wgpu
import wgpu.utils
from wgpu.utils import get_default_device
import time
import shutil


def get_terminal_size() -> tuple[int, int]:
    """
    Get the size of the terminal window.

    Returns:
        tuple: The number of columns and lines in the terminal.
    """
    size = shutil.get_terminal_size()
    return size.columns, size.lines


def pack_rgba_to_uint32(rgba_array: np.ndarray) -> np.ndarray:
    """
    Convert a NumPy array of shape (H, W, 4) representing RGBA values (uint8)
    into a single uint32 array of shape (H, W).

    Args:
        rgba_array (np.ndarray): The input RGBA array.

    Returns:
        np.ndarray: The packed uint32 array.
    """
    rgba_array = rgba_array.astype(np.uint8)
    packed_array = (
        (rgba_array[..., 0].astype(np.uint32) << 24)
        | (rgba_array[..., 1].astype(np.uint32) << 16)
        | (rgba_array[..., 2].astype(np.uint32) << 8)
        | (rgba_array[..., 3].astype(np.uint32))
    )
    return packed_array


def clear_terminal() -> None:
    """
    Clear the terminal using ANSI escape codes.
    """
    print("\033[2J\033[H", end="", flush=True)


def print_high_res_image(image: np.ndarray) -> None:
    """
    Print a high-resolution image using half-block characters for better detail.

    Args:
        image (np.ndarray): The input image array.
    """
    height, width = image.shape
    for y in range(0, height, 2):
        for x in range(width):
            top_pixel = image[y, x] if y < height else 0
            bottom_pixel = image[y + 1, x] if y + 1 < height else 0
            r1, g1, b1 = (
                (top_pixel >> 24) & 0xFF,
                (top_pixel >> 16) & 0xFF,
                (top_pixel >> 8) & 0xFF,
            )
            r2, g2, b2 = (
                (bottom_pixel >> 24) & 0xFF,
                (bottom_pixel >> 16) & 0xFF,
                (bottom_pixel >> 8) & 0xFF,
            )
            print(f"\033[38;2;{r1};{g1};{b1}m\033[48;2;{r2};{g2};{b2}m▄\033[0m", end="")
        print()


def swap_alternate_rows(arr: np.ndarray) -> np.ndarray:
    """
    Swap every other row of a NumPy array in place.

    Args:
        arr (np.ndarray): Input array (2D or higher).

    Returns:
        np.ndarray: Array with alternate rows swapped.
    """
    if arr.shape[0] < 2:
        return arr
    arr[0::2], arr[1::2] = arr[1::2], arr[0::2].copy()
    return arr


def init_buffers(device: wgpu.GPUDevice, vertices: np.ndarray) -> wgpu.GPUBuffer:
    """
    Initialize the vertex buffer and a copy buffer.

    Args:
        device (wgpu.GPUDevice): The GPU device.
        vertices (np.ndarray): The vertex data.

    Returns:
        wgpu.GPUBuffer: The initialized vertex buffer.
    """
    # Create the vertex buffer
    vertex_buffer = device.create_buffer(
        size=vertices.nbytes,
        usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
    )

    # Create a copy buffer to update the vertex buffer
    vertex_buffer.copy_buffer = device.create_buffer(
        size=vertices.nbytes,
        usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
    )

    return vertex_buffer


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
    # note np buffer is in uint8 format and has shape (height, width, 4) rows cols channels
    buffer = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 4))
    readback_buffer.unmap()

    return buffer


def rotate_vertices(
    vertices: np.ndarray,
    angle: float,
    vertex_buffer: wgpu.GPUBuffer,
    device: wgpu.GPUDevice,
) -> np.ndarray:
    """
    Rotate the vertices around the Z-axis by the given angle.

    Args:
        vertices (np.ndarray): The vertex data.
        angle (float): The rotation angle in radians.
        vertex_buffer (wgpu.GPUBuffer): The vertex buffer.
        device (wgpu.GPUDevice): The GPU device.

    Returns:
        np.ndarray: The rotated vertex data.
    """
    rotation_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    rotated_vertices = vertices.copy()
    rotated_vertices[:, :3] = np.dot(vertices[:, :3], rotation_matrix.T)
    tmp_buffer = vertex_buffer.copy_buffer
    tmp_buffer.map_sync(wgpu.MapMode.WRITE)
    tmp_buffer.write_mapped(rotated_vertices.tobytes())
    tmp_buffer.unmap()
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(
        tmp_buffer, 0, vertex_buffer, 0, rotated_vertices.nbytes
    )
    device.queue.submit([command_encoder.finish()])


def main() -> None:
    """
    Main function to render a rotating triangle and print the image to the terminal
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
    vertex_buffer = init_buffers(device, vertices)
    angle = 0.0

    try:
        while True:
            clear_terminal()
            rotate_vertices(vertices, angle, vertex_buffer, device)
            texture = render_triangle(device, pipeline, vertex_buffer, WIDTH, HEIGHT)
            buffer = copy_texture_to_buffer(device, texture, WIDTH, HEIGHT)
            buffer = pack_rgba_to_uint32(buffer)
            buffer = swap_alternate_rows(buffer)
            print_high_res_image(buffer)

            angle += 0.2
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Clean up resources just in case.
        texture.destroy()
        vertex_buffer.copy_buffer.destroy()
        vertex_buffer.destroy()
        device.destroy()

        pass


if __name__ == "__main__":
    cols, rows = get_terminal_size()

    WIDTH = 64
    HEIGHT = rows
    main()
