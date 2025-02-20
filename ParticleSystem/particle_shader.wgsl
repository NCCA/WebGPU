@group(0) @binding(0) var<uniform> uniforms : Uniforms;
struct Uniforms
{
    MVP : mat4x4<f32>,
};

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) fragColor: vec3<f32>,
};

@vertex
fn vertex_main(input: VertexIn) -> VertexOut {
    var output: VertexOut;
    output.position = uniforms.MVP * vec4<f32>(input.position, 1.0);
    output.fragColor = input.color;
    return output;
}
@fragment
fn fragment_main(@location(0) fragColor: vec3<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(fragColor, 1.0); // Simple color output
}
