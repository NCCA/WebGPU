"""
This is a static class to store and manage the pipelines used in WebGPU, to start with there
will be a few common pipelines and then the ability for the user to add their own

As pipelines are closely related to the shaders some of the shader code is included here
as well, think most of the shaders will be stored in their own python file as strings and imported
to use them. Most of the shaders are quite simple in this case.
"""
import wgpu
import numpy as np
import nccapy
from PipelineShaders import line_shader, diffuse_shader

_default_depth_stencil = {
    "format": wgpu.TextureFormat.depth24plus,
    "depth_write_enabled": True,
    "depth_compare": wgpu.CompareFunction.less,
}

_default_multisample = {"count": 1, "mask": 0xFFFFFFFF, "alpha_to_coverage_enabled": False}
_default_layout = "auto"

_primitive_line_list  = {
            "topology": wgpu.PrimitiveTopology.line_list,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        }

_primitive_triangle_list  = {
            "topology": wgpu.PrimitiveTopology.triangle_list,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        }

_primitive_triangle_strip  = {
            "topology": wgpu.PrimitiveTopology.triangle_strip,
            "front_face": wgpu.FrontFace.ccw,
            "cull_mode": wgpu.CullMode.none,
        }



_FLOAT_SIZE = np.dtype(np.float32).itemsize

class PipeLineException(Exception):
    pass


class PipelineNotFound(PipeLineException):
    pass


class _pipelineEntry:
    def __init__(self, pipeline, bind_group, uniform_buffer, uniform_data):
        self.pipeline = pipeline
        self.bind_group = bind_group
        self.uniform_buffer = uniform_buffer
        self.uniform_data = uniform_data


class Pipelines:
    _pipelines = {}

    @classmethod
    def get_pipeline(cls, name):
        if name in cls._pipelines:
            return cls._pipelines[name]
        else:
            raise PipelineNotFound(f"Pipeline {name} not found")

    @classmethod
    def create_line_pipeline(cls, name, device):
        shader = device.create_shader_module(code=line_shader)
        label = "line_pipeline"
        vertex = {
            "module": shader,
            "entry_point": "vertex_main",
            "buffers": [
                {
                    "array_stride": 3 * _FLOAT_SIZE, # 3 floats * 4 bytes
                    "attributes": [
                        {"shader_location": 0, "offset": 0, "format": "float32x3"},
                    ],
                }
            ],
        }
        fragment = {
            "module": shader,
            "entry_point": "fragment_main",
            "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
        }
        
        pipeline = device.create_render_pipeline(
            label=label,
            layout=_default_layout,
            vertex=vertex,
            fragment=fragment,
            primitive=_primitive_line_list,
            depth_stencil=_default_depth_stencil,
            multisample=_default_multisample,
        )

        # Create a uniform buffer
        uniform_data = np.zeros(
            (),
            dtype=[
                ("MVP", "float32", (16)),
                ("colour", "float32", (3)),
                ("padding", "float32", (1)),  # to 80 bytes
            ],
        )

        uniform_buffer = device.create_buffer_with_data(
            data=uniform_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="line_pipeline_uniform_buffer",
        )

        bind_group_layout = pipeline.get_bind_group_layout(0)
        # Create the bind group
        bind_group = device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,  # Matches @binding(0) in the shader
                    "resource": {"buffer": uniform_buffer},
                }
            ],
        )

        pipeline_entry = _pipelineEntry(
            pipeline=pipeline,
            bind_group=bind_group,
            uniform_buffer=uniform_buffer,
            uniform_data=uniform_data,
        )
        cls._pipelines[name] = pipeline_entry
        return pipeline_entry

    @classmethod
    def create_diffuse_pipeline(cls, name, device):
        shader = device.create_shader_module(code=diffuse_shader)
        label = "diffuse_triangle_pipeline"
        vertex = {
            "module": shader,
            "entry_point": "vertex_main",
            "buffers": [
                {
                    "array_stride": 8  * _FLOAT_SIZE, #x,y,z nx,ny,nz,u,v
                    "attributes": [
                        {"shader_location": 0, "offset": 0*_FLOAT_SIZE, "format": "float32x3"},
                        {"shader_location": 1, "offset": 3*_FLOAT_SIZE, "format": "float32x3"},
                        {"shader_location": 2, "offset": 6*_FLOAT_SIZE, "format": "float32x2"},
                    ],
                }
            ],
        }
        fragment = {
            "module": shader,
            "entry_point": "fragment_main",
            "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
        }
        
        pipeline = device.create_render_pipeline(
            label=label,
            layout=_default_layout,
            vertex=vertex,
            fragment=fragment,
            primitive=_primitive_triangle_strip,
            depth_stencil=_default_depth_stencil,
            multisample=_default_multisample,
        )

        # Create a uniform buffer
        vertex_uniform_data = np.zeros(
            (),
            dtype=[
                ("MVP", "float32", (16)),
                ("model_view", "float32", (16)),
                ("normal_matrix", "float32", (16)), # need 4x4 for mat3 
                ("colour", "float32", (3)),
                ("padding", "float32", (4)),  # to 208
                ],
        )

        vertex_uniform_buffer = device.create_buffer_with_data(
            data=vertex_uniform_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="vertex_uniform_data",
        )

        # Create a uniform buffer
        light_uniform_data = np.zeros(
            (),
            dtype=[
                ("light_pos", "float32", (3)),
                ("light_diffuse", "float32", (3)),
                ("padding", "float32", (8)),  # to 32 bytes
                ],
        )

        light_uniform_buffer = device.create_buffer_with_data(
            data=light_uniform_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="light_uniform_data",
        )

        bind_group_layout_0 = pipeline.get_bind_group_layout(0)
        bind_group_layout_1 = pipeline.get_bind_group_layout(1)
        # Create the bind group
        bind_group_0 = device.create_bind_group(
            layout=bind_group_layout_0,
            entries=[
                {
                    "binding": 0,  # Matches @binding(0) in the shader
                    "resource": {"buffer": vertex_uniform_buffer},
                }
            ],
        )

        bind_group_1 = device.create_bind_group(
            layout=bind_group_layout_1,
            entries=[
                {
                    "binding": 0,  # Matches @binding(0) in the shader
                    "resource": {"buffer": light_uniform_buffer},
                }
            ],
        )

        pipeline_entry = _pipelineEntry(
            pipeline=pipeline,
            bind_group=[bind_group_0, bind_group_1],
            uniform_buffer=[vertex_uniform_buffer, light_uniform_buffer],
            uniform_data=[vertex_uniform_data, light_uniform_data],
        )
        cls._pipelines[name] = pipeline_entry
        return pipeline_entry

