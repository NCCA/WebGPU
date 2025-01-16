# WebGPU Code

Work in progress python / WebGPU demos for next academic year.

Need to install at least wgpu numpy and PySide6 nccapy



# Design Considerations

## pipelines

When generating a pipeline to work a bit like OpenGL we need to know in advance how many meshes we are going to render. This is because we need to generate a buffer for all the unique uniforms. It is typically better to have a single buffer for all the uniforms and then use the offset to access the correct uniform. This is because the GPU is optimized for reading memory in blocks.




