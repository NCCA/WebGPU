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
from PipelineShaders import line_shader


class PipeLineException(Exception):
    pass
class PipelineNotFound(PipeLineException):
    pass


class _pipelineEntry:
    def __init__(self,pipeline,bind_group,uniform_buffer,uniform_data):
        self.pipeline = pipeline
        self.bind_group = bind_group
        self.uniform_buffer = uniform_buffer
        self.uniform_data = uniform_data

class Pipelines :
    _pipelines = {}

    @classmethod
    def get_pipeline(cls,name):
        if name in cls._pipelines:
            return cls._pipelines[name]
        else:
            raise PipelineNotFound(f"Pipeline {name} not found")

    @classmethod
    def create_line_pipeline(cls,name,device):
        shader = device.create_shader_module(code=line_shader)
        pipeline = device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": shader,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 3 * 4,
                        "attributes": [
                            {"shader_location": 0, "offset": 0, "format": "float32x3"},
                            {"shader_location": 1, "offset": 0, "format": "float32x3"},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader,
                "entry_point": "fragment_main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.rgba8unorm,
                    }
                ],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.line_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": 1,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
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

        uniform_data["MVP"] = nccapy.Mat4().get_numpy().flatten()
        uniform_data["colour"] = np.array([1.0, 1.0, 0.0])
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

        pipeline_entry = _pipelineEntry(pipeline=pipeline,bind_group=bind_group,uniform_buffer=uniform_buffer,uniform_data=uniform_data)
        cls._pipelines[name] = pipeline_entry
        return pipeline_entry
