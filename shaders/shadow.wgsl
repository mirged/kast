@group(0) @binding(0) var<uniform> shadow_matrix: mat4x4<f32>;
@group(1) @binding(0) var<storage, read> objects: array<ObjectData>;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(4) obj_index: u32,
) -> @builtin(position) vec4<f32> {
    let obj = objects[obj_index];
    return shadow_matrix * obj.model * vec4<f32>(position, 1.0);
}
