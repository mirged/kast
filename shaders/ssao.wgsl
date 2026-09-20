struct SsaoUniform {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    params: vec4<f32>,
    screen: vec4<f32>,
};

@group(0) @binding(0) var<uniform> ssao: SsaoUniform;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler: sampler;
@group(0) @binding(4) var ao_tex: texture_2d<f32>;

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn sample_kernel(i: u32, n: u32) -> vec3<f32> {
    let xi = hammersley(i, n);
    let theta = acos(1.0 - xi.y);
    let phi = 2.0 * PI * xi.x;
    return vec3<f32>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
}

@fragment
fn fs_ssao(in: FullscreenOut) -> @location(0) vec4<f32> {
    let depth = textureSampleLevel(depth_tex, tex_sampler, in.uv, 0.0);
    if depth >= 1.0 {
        return vec4<f32>(1.0, 0.0, 0.0, 0.0);
    }

    let view_pos = reconstruct_view_pos(in.uv, depth, ssao.inv_proj);
    let world_normal = textureSampleLevel(normal_tex, tex_sampler, in.uv, 0.0).xyz;
    let view_normal = normalize((ssao.view * vec4<f32>(world_normal, 0.0)).xyz);

    let angle = hash12(in.uv * ssao.screen.xy) * 2.0 * PI;
    let random_vec = vec3<f32>(cos(angle), sin(angle), 0.0);
    var tangent = random_vec - view_normal * dot(random_vec, view_normal);
    if length(tangent) < 1e-4 {
        tangent = vec3<f32>(1.0, 0.0, 0.0);
    }
    tangent = normalize(tangent);
    let bitangent = cross(view_normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, view_normal);

    let radius = ssao.params.x;
    let bias = ssao.params.y;
    let sample_count = u32(max(ssao.screen.x, 1.0));

    var occlusion = 0.0;
    let count = f32(sample_count);
    for (var i = 0u; i < sample_count; i = i + 1u) {
        let kernel = sample_kernel(i, sample_count);
        let scale = mix(0.15, 1.0, f32(i) / count) * radius;
        let sample_pos = view_pos + tbn * (kernel * scale);

        let clip = ssao.proj * vec4<f32>(sample_pos, 1.0);
        let ndc = clip.xyz / clip.w;
        let sample_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);

        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }

        let sample_depth = textureSampleLevel(depth_tex, tex_sampler, sample_uv, 0.0);
        let stored_z = -reconstruct_view_pos(sample_uv, sample_depth, ssao.inv_proj).z;
        let sample_z = -sample_pos.z;

        let range_check = smoothstep(0.0, 1.0, radius / max(abs(stored_z - sample_z), 1e-4));
        if sample_z > stored_z + bias {
            occlusion += range_check;
        }
    }

    occlusion = 1.0 - (occlusion / count) * ssao.params.z;
    occlusion = pow(clamp(occlusion, 0.0, 1.0), ssao.params.w);
    return vec4<f32>(occlusion, 0.0, 0.0, 0.0);
}

@fragment
fn fs_blur(in: FullscreenOut) -> @location(0) vec4<f32> {
    let texel = ssao.screen.zw;
    let depth = textureSampleLevel(depth_tex, tex_sampler, in.uv, 0.0);
    let center_z = -reconstruct_view_pos(in.uv, depth, ssao.inv_proj).z;

    var sum = 0.0;
    var weight = 0.0;
    for (var y = -2; y <= 2; y = y + 1) {
        for (var x = -2; x <= 2; x = x + 1) {
            let uv = in.uv + vec2<f32>(f32(x), f32(y)) * texel;
            let d = textureSampleLevel(depth_tex, tex_sampler, uv, 0.0);
            let z = -reconstruct_view_pos(uv, d, ssao.inv_proj).z;
            let w = 1.0 / (1.0 + abs(z - center_z) * 8.0);
            let ao = textureSampleLevel(ao_tex, tex_sampler, uv, 0.0).r;
            sum += ao * w;
            weight += w;
        }
    }
    return vec4<f32>(sum / max(weight, 1e-4), 0.0, 0.0, 0.0);
}
