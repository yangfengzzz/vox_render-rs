[[block]]
struct Uniforms {
    view_proj: mat4x4<f32>;
    model: mat4x4<f32>;
};
[[group(1), binding(0)]]
var uniforms: Uniforms;

struct VertexInput {
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] tex_coord: vec2<f32>;
    [[location(2)]] normal: vec3<f32>;
};

struct VertexOutput {
    [[location(0)]] tex_coord: vec2<f32>;
    [[builtin(position)]] position: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coord = model.tex_coord;
    out.position =  uniforms.view_proj * uniforms.model * vec4<f32>(model.position, 1.0);
    return out;
}

//-----------------------------------------------------------------------------
[[group(0), binding(0)]]
var r_color: texture_2d<u32>;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let tex = textureLoad(r_color, vec2<i32>(in.tex_coord * 256.0), 0);
    let v = f32(tex.x) / 255.0;
    return vec4<f32>(1.0 - (v * 5.0), 1.0 - (v * 15.0), 1.0 - (v * 50.0), 1.0);
}

[[stage(fragment)]]
fn fs_wire() -> [[location(0)]] vec4<f32> {
    return vec4<f32>(0.0, 0.5, 0.0, 0.5);
}