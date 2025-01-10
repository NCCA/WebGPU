
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct Uniforms 
{
    MVP : mat4x4<f32>,
    vertex_colour : vec3<f32>
};

struct VertexInput 
{
    @location(0) position : vec3<f32>,
};

struct VertexOutput 
{
    @builtin(position) position : vec4<f32>,
    @location(0) colour : vec3<f32>
};



@vertex
fn vertex_main(input : VertexInput) -> VertexOutput 
{
    var output : VertexOutput;
    output.position = uniforms.MVP * vec4<f32>(input.position, 1.0);
    output.colour = uniforms.vertex_colour;
    return output;
}

struct FragmentInput 
{
    @location(0) colour : vec3<f32>
};

struct FragmentOutput 
{
    @location(0) colour : vec4<f32>
};

@fragment
fn fragment_main(input : FragmentInput) -> FragmentOutput 
{
    var output : FragmentOutput;
    output.colour = vec4<f32>(input.colour, 1.0);
    return output;
}
