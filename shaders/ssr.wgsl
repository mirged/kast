struct SsrUniform {
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    params: vec4<f32>,
    screen: vec4<f32>,
};

@group(0) @binding(0) var<uniform> ssr: SsrUniform;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var normal_tex: texture_2d<f32>;
@group(0) @binding(4) var tex_sampler: sampler;

fn hash12(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4<f32> {
    let source = textureSampleLevel(hdr_tex, tex_sampler, in.uv, 0.0);
    let depth = textureSampleLevel(depth_tex, tex_sampler, in.uv, 0.0);
    if depth <= 0.0 {
        return vec4<f32>(source.rgb, 1.0);
    }

    let view_pos = reconstruct_view_pos(in.uv, depth, ssr.inv_proj);
    let normal_sample = textureSampleLevel(normal_tex, tex_sampler, in.uv, 0.0);
    let world_normal = normal_sample.xyz;
    let roughness = normal_sample.w;
    let view_normal = normalize((ssr.view * vec4<f32>(world_normal, 0.0)).xyz);
    let view_dir = normalize(view_pos);
    var reflect_dir = reflect(view_dir, view_normal);
    if reflect_dir.z > 0.0 {
        reflect_dir.z = -reflect_dir.z;
    }

    let max_steps = i32(ssr.params.x);
    let thickness = ssr.params.y;
    let stride = ssr.params.z;
    let intensity = ssr.params.w;
    let max_distance = ssr.screen.w;

    let jitter = hash12(in.uv * ssr.screen.xy + ssr.screen.z);
    var ray = view_pos + view_normal * 0.03;

    var hit_uv = vec2<f32>(0.0);
    var hit = false;
    var travelled = jitter * stride;

    for (var i = 0; i < max_steps; i = i + 1) {
        travelled += stride * (1.0 + f32(i) * 0.12);
        if travelled > max_distance {
            break;
        }
        ray = view_pos + reflect_dir * travelled;

        let clip = ssr.proj * vec4<f32>(ray, 1.0);
        if clip.w <= 0.0 {
            break;
        }
        let ndc = clip.xyz / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            break;
        }

        let scene_depth = textureSampleLevel(depth_tex, tex_sampler, uv, 0.0);
        let surface_z = -reconstruct_view_pos(uv, scene_depth, ssr.inv_proj).z;
        let ray_z = -ray.z;

        let diff = ray_z - surface_z;
        if diff > 0.0 && diff < thickness {
            hit = true;
            hit_uv = uv;
            break;
        }
    }

    var reflection = vec3<f32>(0.0);
    if hit {
        let gloss = 1.0 - smoothstep(0.05, 0.5, roughness);
        let ndotv = max(dot(-view_dir, view_normal), 0.0);
        let fresnel = pow(1.0 - ndotv, 4.0) * 0.9 + 0.1;
        let edge = smoothstep(0.0, 0.12, hit_uv.x) * smoothstep(1.0, 0.88, hit_uv.x)
            * smoothstep(0.0, 0.12, hit_uv.y) * smoothstep(1.0, 0.88, hit_uv.y);
        reflection = textureSampleLevel(hdr_tex, tex_sampler, hit_uv, 0.0).rgb
            * intensity * gloss * fresnel * edge;
    }

    return vec4<f32>(source.rgb + reflection, 1.0);
}
