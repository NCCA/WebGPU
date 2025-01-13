"""
Shaders used in the Pipeline class.
"""

# Line shader with MVP and vertex colour

line_shader = """
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
"""

diffuse_shader = """
@group(0) @binding(0) var<uniform> vertexUniforms : VertexUniforms;
@group(1) @binding(0) var<uniform> lightUniforms : LightUniforms;

struct VertexUniforms 
{
    MVP : mat4x4<f32>,
    model_view : mat4x4<f32>,
    normal_matrix : mat4x4<f32>,
    colour : vec3<f32>,
};

struct LightUniforms
{
    light_pos : vec3<f32>,
    light_diffuse : vec3<f32>,
};


struct VertexInput 
{
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>
};

struct VertexOutput 
{
    @builtin(position) position : vec4<f32>,
    @location(0) normal : vec3<f32>,
    @location(1) uv : vec2<f32>,
    @location(2) frag_pos : vec3<f32>

};

fn extract_mat3x3(mat: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        mat[0].xyz,
        mat[1].xyz,
        mat[2].xyz
    );
}

@vertex
fn vertex_main(input : VertexInput) -> VertexOutput 
{
    var output : VertexOutput;
    output.position = vertexUniforms.MVP * vec4<f32>(input.position, 1.0);
    output.normal = extract_mat3x3(vertexUniforms.normal_matrix) * input.normal;
    output.uv = input.uv;
    output.frag_pos = (vertexUniforms.model_view * vec4<f32>(input.position, 1.0)).xyz;
    return output;
}

struct FragmentInput
{
    @location(0) normal : vec3<f32>,
    @location(1) uv : vec2<f32>,
    @location(2) frag_pos : vec3<f32>
    
};

struct FragmentOutput 
{
    @location(0) colour : vec4<f32>
};

@fragment
fn fragment_main(input : FragmentInput) -> FragmentOutput 
{
    var output : FragmentOutput;
    let N = normalize(input.normal);
    let L = normalize(lightUniforms.light_pos - input.frag_pos);
    let diffuse = max(dot(N,L), 0.0);

    output.colour  += vec4<f32>(vertexUniforms.colour *lightUniforms.light_diffuse *diffuse, 1.0);
    return output;
}

"""
