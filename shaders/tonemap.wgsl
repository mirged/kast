struct PostUniform {
    params: vec4<f32>,
    modes: vec4<f32>,
    screen: vec4<f32>,
};

@group(0) @binding(0) var<uniform> post: PostUniform;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler: sampler;

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4<f32> {
    var color = textureSample(hdr_tex, tex_sampler, in.uv).rgb;

    if post.modes.y > 0.5 {
        color = color + textureSample(bloom_tex, tex_sampler, in.uv).rgb * post.params.z;
    }

    color = color * post.params.x;
    color = tonemap(color, u32(post.modes.x));

    if post.modes.w > 0.5 {
        color = linear_to_srgb(color);
    }

    let centered = in.uv - 0.5;
    let vig = 1.0 - post.params.w * dot(centered, centered) * 2.0;
    color = color * clamp(vig, 0.0, 1.0);

    return vec4<f32>(color, 1.0);
}
