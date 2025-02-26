@group(0) @binding(0) var<uniform> uniforms : Uniforms;
struct Uniforms
{
    MVP : mat4x4<f32>,
    eye : vec3<f32>,
    circle_square : i32
};

struct VertexIn {
    @location(0) quad_pos: vec2<f32>,
    @location(1) position: vec4<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) fragColor: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexIn) -> VertexOut
 {
    var output: VertexOut;



    var cam_up: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

    let to_camera = normalize(uniforms.eye - input.position.xyz);
    let cam_right = cross(to_camera, cam_up);

    
    let size = input.position.w;
    
    let tx_pos=input.position.xyz;

    let world_pos = tx_pos + cam_right * input.quad_pos.x * size + cam_up * input.quad_pos.y * size;

    output.uv = input.quad_pos.xy+ vec2<f32>(1.0, 1.0) * 0.5; // Convert from [-1,1] to [0,1]



    output.position = uniforms.MVP * vec4<f32>(world_pos, 1.0);
    
    output.fragColor = input.color;
    return output;
}

@fragment
fn fragment_main(
    @location(0) fragColor: vec4<f32> , 
    @location(1) uv: vec2<f32>
) -> @location(0) vec4<f32> 
{
    if (uniforms.circle_square == 0) {
        return vec4<f32>(fragColor); // Simple color output
    }
    let center = vec2<f32>(0.5, 0.5); // Center of the quad in UV space
    let dist = length(uv - center); // Distance from center
    let radius = 0.5; // Circle radius (quad is 1.0 in UV space)

    if (dist > radius) {
        discard; // Remove pixels outside the circle
    }


    return vec4<f32>(fragColor); // Simple color output
}
