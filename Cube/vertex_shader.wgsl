
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct Uniforms 
{
    mvpMatrix : mat4x4<f32>
};

struct VertexInput 
{
    @location(0) position : vec3<f32>,
    @location(1) color : vec3<f32>
};

struct VertexOutput 
{
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec3<f32>
};



@vertex
fn main(input : VertexInput) -> VertexOutput 
{
    var output : VertexOutput;
    output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 1.0);
    output.color = input.color;
    return output;
}