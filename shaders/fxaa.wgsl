struct FxaaUniform {
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> fxaa: FxaaUniform;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

fn fxaa_sample(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(color_tex, tex_sampler, uv, 0.0);
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4<f32> {
    let texel = fxaa.params.xy;
    let center = fxaa_sample(in.uv);

    if fxaa.params.z < 0.5 {
        return center;
    }

    let nw = fxaa_sample(in.uv + vec2<f32>(-1.0, -1.0) * texel).rgb;
    let ne = fxaa_sample(in.uv + vec2<f32>(1.0, -1.0) * texel).rgb;
    let sw = fxaa_sample(in.uv + vec2<f32>(-1.0, 1.0) * texel).rgb;
    let se = fxaa_sample(in.uv + vec2<f32>(1.0, 1.0) * texel).rgb;

    let luma_m = luma(center.rgb);
    let luma_nw = luma(nw);
    let luma_ne = luma(ne);
    let luma_sw = luma(sw);
    let luma_se = luma(se);

    let luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    let luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    let range = luma_max - luma_min;
    if range < max(0.0312, luma_max * 0.125) {
        return center;
    }

    let dir = vec2<f32>(
        -((luma_nw + luma_ne) - (luma_sw + luma_se)),
        ((luma_nw + luma_sw) - (luma_ne + luma_se)),
    );
    let dir_reduce = vec2<f32>(
        1.0 / (min(abs(dir.x), abs(dir.y)) + max((luma_nw + luma_ne + luma_sw + luma_se) * 0.03125, 0.0078125)),
        1.0 / (min(abs(dir.x), abs(dir.y)) + max((luma_nw + luma_ne + luma_sw + luma_se) * 0.03125, 0.0078125)),
    );
    let dir2 = clamp(dir * dir_reduce, vec2<f32>(-8.0), vec2<f32>(8.0)) * texel;

    let rgb_a = 0.5 * (
        fxaa_sample(in.uv + dir2 * (1.0 / 3.0 - 0.5)).rgb +
        fxaa_sample(in.uv + dir2 * (2.0 / 3.0 - 0.5)).rgb
    );
    let rgb_b = rgb_a * 0.5 + 0.25 * (
        fxaa_sample(in.uv + dir2 * -0.5).rgb +
        fxaa_sample(in.uv + dir2 * 0.5).rgb
    );

    let luma_b = luma(rgb_b);
    if luma_b < luma_min || luma_b > luma_max {
        return vec4<f32>(rgb_a, center.a);
    }
    return vec4<f32>(rgb_b, center.a);
}
