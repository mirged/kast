struct DofUniform {
    inv_proj: mat4x4<f32>,
    params: vec4<f32>,
    depth: vec4<f32>,
};

@group(0) @binding(0) var<uniform> dof: DofUniform;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var tex_sampler: sampler;

fn linear_at(uv: vec2<f32>) -> f32 {
    let d = textureSampleLevel(depth_tex, tex_sampler, uv, 0.0);
    if d >= 1.0 {
        return 1e6;
    }
    return -reconstruct_view_pos(uv, d, dof.inv_proj).z;
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4<f32> {
    let center = textureSampleLevel(hdr_tex, tex_sampler, in.uv, 0.0).rgb;
    let center_depth = linear_at(in.uv);
    let focus = dof.params.x;
    let range = max(dof.params.y, 1e-3);
    let max_radius = dof.params.z;
    let aperture = dof.params.w;

    let center_coc = clamp(abs(center_depth - focus) / range, 0.0, 1.0);
    let radius = center_coc * max_radius * aperture;
    if radius < 0.5 {
        return vec4<f32>(center, 1.0);
    }

    let texel = dof.depth.zw;
    let sample_count = 32;
    var sum = center;
    var weight = 1.0;

    for (var i = 0; i < sample_count; i = i + 1) {
        let t = (f32(i) + 0.5) / f32(sample_count);
        let r = sqrt(t) * radius;
        let angle = f32(i) * 2.39996323 + center_coc * 6.2831853;
        let offset = vec2<f32>(cos(angle), sin(angle)) * r * texel;
        let uv = in.uv + offset;

        let sample_depth = linear_at(uv);
        let sample_coc = clamp(abs(sample_depth - focus) / range, 0.0, 1.0) * max_radius * aperture;
        let is_background = step(1e5, sample_depth);
        let w = select(step(r, sample_coc), 1.0, is_background > 0.5);

        sum += textureSampleLevel(hdr_tex, tex_sampler, uv, 0.0).rgb * w;
        weight += w;
    }

    return vec4<f32>(sum / weight, 1.0);
}
