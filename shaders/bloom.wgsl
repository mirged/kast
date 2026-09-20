struct BloomUniform {
    params: vec4<f32>,
    screen: vec4<f32>,
};

@group(0) @binding(0) var<uniform> bloom: BloomUniform;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;

@fragment
fn fs_prefilter(in: FullscreenOut) -> @location(0) vec4<f32> {
    let c = textureSample(src_tex, tex_sampler, in.uv).rgb;
    let threshold = bloom.params.z;
    let knee = bloom.params.w;
    let brightness = max(c.r, max(c.g, c.b));
    var soft = brightness - threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = (soft * soft) / (4.0 * knee + 1e-4);
    let contribution = max(soft, brightness - threshold) / max(brightness, 1e-4);
    return vec4<f32>(c * contribution, 1.0);
}

@fragment
fn fs_downsample(in: FullscreenOut) -> @location(0) vec4<f32> {
    let texel = bloom.params.xy;
    let uv = in.uv;
    let a = textureSample(src_tex, tex_sampler, uv + vec2<f32>(-1.0, -1.0) * texel).rgb;
    let b = textureSample(src_tex, tex_sampler, uv + vec2<f32>(1.0, -1.0) * texel).rgb;
    let c = textureSample(src_tex, tex_sampler, uv + vec2<f32>(-1.0, 1.0) * texel).rgb;
    let d = textureSample(src_tex, tex_sampler, uv + vec2<f32>(1.0, 1.0) * texel).rgb;
    let center = textureSample(src_tex, tex_sampler, uv).rgb;
    return vec4<f32>((a + b + c + d + center * 2.0) / 6.0, 1.0);
}

@fragment
fn fs_upsample(in: FullscreenOut) -> @location(0) vec4<f32> {
    let texel = bloom.params.xy;
    let uv = in.uv;
    var sum = vec3<f32>(0.0);
    sum += textureSample(src_tex, tex_sampler, uv + vec2<f32>(-1.0, -1.0) * texel).rgb;
    sum += textureSample(src_tex, tex_sampler, uv + vec2<f32>(0.0, -1.0) * texel).rgb * 2.0;
    sum += textureSample(src_tex, tex_sampler, uv + vec2<f32>(1.0, -1.0) * texel).rgb;
    sum += textureSample(src_tex, tex_sampler, uv + vec2<f32>(-1.0, 0.0) * texel).rgb * 2.0;
    sum += textureSample(src_tex, tex_sampler, uv).rgb * 4.0;
    sum += textureSample(src_tex, tex_sampler, uv + vec2<f32>(1.0, 0.0) * texel).rgb * 2.0;
    sum += textureSample(src_tex, tex_sampler, uv + vec2<f32>(-1.0, 1.0) * texel).rgb;
    sum += textureSample(src_tex, tex_sampler, uv + vec2<f32>(0.0, 1.0) * texel).rgb * 2.0;
    sum += textureSample(src_tex, tex_sampler, uv + vec2<f32>(1.0, 1.0) * texel).rgb;
    return vec4<f32>(sum / 16.0, 1.0);
}
