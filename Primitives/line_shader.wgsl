
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct Uniforms 
{
    mvpMatrix : mat4x4<f32>
};

struct VertexInput 
{
    @location(0) position : vec3<f32>,
};

struct VertexOutput 
{
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec3<f32>
};



@vertex
fn vertex_main(input : VertexInput) -> VertexOutput 
{
    var output : VertexOutput;
    output.position = uniforms.mvpMatrix * vec4<f32>(input.position, 1.0);
    output.color = vec3<f32>(1.0, 0.0, 0.0);
    return output;
}

struct FragmentInput 
{
    @location(0) color : vec3<f32>
};

struct FragmentOutput 
{
    @location(0) color : vec4<f32>
};

@fragment
fn fragment_main(input : FragmentInput) -> FragmentOutput 
{
    var output : FragmentOutput;
    output.color = vec4<f32>(input.color, 1.0);
    return output;
}
