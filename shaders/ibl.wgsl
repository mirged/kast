struct SkyUniform {
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    params: vec4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    zenith: vec4<f32>,
    horizon: vec4<f32>,
};

struct IblUniform {
    params: vec4<f32>,
    face: vec4<u32>,
};

@group(0) @binding(0) var<uniform> sky: SkyUniform;
@group(0) @binding(1) var env_tex: texture_cube<f32>;
@group(0) @binding(2) var env_sampler: sampler;
@group(0) @binding(3) var<uniform> ibl: IblUniform;

fn face_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let a = uv * 2.0 - 1.0;
    let s = a.x;
    let t = a.y;
    switch face {
        case 0u: { return normalize(vec3<f32>(1.0, -t, -s)); }
        case 1u: { return normalize(vec3<f32>(-1.0, -t, s)); }
        case 2u: { return normalize(vec3<f32>(s, 1.0, t)); }
        case 3u: { return normalize(vec3<f32>(s, -1.0, -t)); }
        case 4u: { return normalize(vec3<f32>(s, -t, 1.0)); }
        default: { return normalize(vec3<f32>(-s, -t, -1.0)); }
    }
}

@fragment
fn fs_env(in: FullscreenOut) -> @location(0) vec4<f32> {
    let dir = face_dir(ibl.face.x, in.uv);
    let c = sky_color_ext(dir, normalize(sky.sun_direction.xyz), sky.sun_color.rgb * sky.sun_color.a, sky.zenith.rgb, sky.horizon.rgb, sky.zenith.a);
    return vec4<f32>(c, 1.0);
}

@fragment
fn fs_irradiance(in: FullscreenOut) -> @location(0) vec4<f32> {
    let n = face_dir(ibl.face.x, in.uv);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if abs(n.y) > 0.999 {
        up = vec3<f32>(0.0, 0.0, 1.0);
    }
    let right = normalize(cross(up, n));
    let up2 = cross(n, right);

    let sample_count = i32(max(ibl.params.z, 1.0));
    let nr = f32(sample_count);
    var irradiance = vec3<f32>(0.0);
    for (var i = 0; i < sample_count; i = i + 1) {
        for (var j = 0; j < sample_count; j = j + 1) {
            let phi = 2.0 * PI * f32(j) / nr;
            let theta = PI * 0.5 * f32(i) / nr;
            let tangent = vec3<f32>(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            let sample_vec = tangent.x * right + tangent.y * up2 + tangent.z * n;
            irradiance += textureSampleLevel(env_tex, env_sampler, sample_vec, 0.0).rgb * cos(theta) * sin(theta);
        }
    }
    irradiance = PI * irradiance / (nr * nr);
    return vec4<f32>(irradiance, 1.0);
}

@fragment
fn fs_prefilter(in: FullscreenOut) -> @location(0) vec4<f32> {
    let n = face_dir(ibl.face.x, in.uv);
    let v = n;
    let roughness = ibl.params.x;
    let sample_count = i32(max(ibl.params.z, 1.0));
    let face_size = ibl.params.w;
    let mip_count = f32(ibl.face.y);

    let sa_texel = 4.0 * PI / (6.0 * face_size * face_size);
    var prefiltered = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 0; i < sample_count; i = i + 1) {
        let xi = hammersley(u32(i), u32(sample_count));
        let h = importance_sample_ggx(xi, n, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);
        let ndotl = max(dot(n, l), 0.0);
        if ndotl > 0.0 {
            let ndoth = max(dot(n, h), 0.0);
            let vdoth = max(dot(v, h), 0.0);
            let d = distribution_ggx(ndoth, roughness);
            let pdf = d * ndoth / max(4.0 * vdoth, 1e-4);
            let sa_sample = 1.0 / max(f32(sample_count) * pdf, 1e-4);
            var mip = 0.0;
            if roughness > 0.0 {
                mip = 0.5 * log2(sa_sample / sa_texel);
            }
            mip = clamp(mip, 0.0, mip_count - 1.0);
            prefiltered += textureSampleLevel(env_tex, env_sampler, l, mip).rgb * ndotl;
            total_weight += ndotl;
        }
    }
    if total_weight > 0.0 {
        prefiltered = prefiltered / total_weight;
    }
    return vec4<f32>(prefiltered, 1.0);
}

@fragment
fn fs_brdf(in: FullscreenOut) -> @location(0) vec4<f32> {
    let ndotv = clamp(in.uv.x, 0.001, 0.999);
    let roughness = clamp(in.uv.y, 0.001, 0.999);
    let v = vec3<f32>(sqrt(1.0 - ndotv * ndotv), 0.0, ndotv);
    let n = vec3<f32>(0.0, 0.0, 1.0);

    var a = 0.0;
    var b = 0.0;
    let sample_count = 256u;
    for (var i = 0u; i < sample_count; i = i + 1u) {
        let xi = hammersley(i, sample_count);
        let h = importance_sample_ggx(xi, n, roughness);
        let l = normalize(2.0 * dot(v, h) * h - v);
        let ndotl = max(l.z, 0.0);
        let ndoth = max(h.z, 0.0);
        let vdoth = max(dot(v, h), 0.0);
        if ndotl > 0.0 {
            let g = geometry_smith(ndotv, ndotl, roughness);
            let g_vis = (g * vdoth) / max(ndoth * ndotv, 1e-4);
            let fc = pow(1.0 - vdoth, 5.0);
            a += (1.0 - fc) * g_vis;
            b += fc * g_vis;
        }
    }
    return vec4<f32>(a / f32(sample_count), b / f32(sample_count), 0.0, 1.0);
}
