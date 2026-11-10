const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 1e-6;

struct Globals {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen: vec4<f32>,
    near_far: vec4<f32>,
    ambient: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    light_meta: vec4<u32>,
    time: vec4<f32>,
    shadow_splits: vec4<f32>,
    shadow_texel: vec4<f32>,
    cascade_matrices: array<mat4x4<f32>, 4>,
};

struct GpuLight {
    position: vec4<f32>,
    direction: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
};

struct ObjectData {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    base_color: vec4<f32>,
    emissive: vec4<f32>,
    material: vec4<f32>,
    uv_transform: vec4<f32>,
    flags0: vec4<u32>,
    flags1: vec4<u32>,
    /// x: detail wavelength (metres, 0 = off), y: strength, z: animate with time
    detail: vec4<f32>,
};

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> FullscreenOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let p = positions[index];
    var out: FullscreenOut;
    out.position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn reconstruct_world_pos(uv: vec2<f32>, depth: f32, inv_view_proj: mat4x4<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = inv_view_proj * ndc;
    return world.xyz / world.w;
}

fn reconstruct_view_pos(uv: vec2<f32>, depth: f32, inv_proj: mat4x4<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let view = inv_proj * ndc;
    return view.xyz / view.w;
}

fn linear_depth(uv: vec2<f32>, depth: f32, inv_proj: mat4x4<f32>) -> f32 {
    let view = reconstruct_view_pos(uv, depth, inv_proj);
    return -view.z;
}

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, EPSILON);
}

fn geometry_schlick_ggx(n_dot_x: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_x / (n_dot_x * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    return f0 + (max(vec3<f32>(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn importance_sample_ggx(xi: vec2<f32>, n: vec3<f32>, roughness: f32) -> vec3<f32> {
    let a = roughness * roughness;
    let phi = 2.0 * PI * xi.x;
    let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let h = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.z) > 0.999);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

fn radical_inverse_vdc(bits_in: u32) -> f32 {
    var bits = bits_in;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return f32(bits) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    return vec2<f32>(f32(i) / f32(n), radical_inverse_vdc(i));
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let cutoff = c < vec3<f32>(0.0031308);
    let low = c * 12.92;
    let high = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, cutoff);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = c < vec3<f32>(0.04045);
    let low = c / 12.92;
    let high = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
    return select(high, low, cutoff);
}

fn aces_narkowicz(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn aces_fitted(x: vec3<f32>) -> vec3<f32> {
    let a = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777),
    );
    let b = mat3x3<f32>(
        vec3<f32>(1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108, 1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605, 1.07602),
    );
    let v = a * x;
    let a2 = v * (v + 0.0245786) - 0.000090537;
    let b2 = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(b * (a2 / b2), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tonemap(x: vec3<f32>, mode: u32) -> vec3<f32> {
    if mode == 0u {
        return aces_narkowicz(x);
    } else if mode == 1u {
        return aces_fitted(x);
    } else if mode == 2u {
        return x / (x + vec3<f32>(1.0));
    }
    return clamp(x, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sky_color(dir_in: vec3<f32>, sun_dir: vec3<f32>, sun_color: vec3<f32>) -> vec3<f32> {
    return sky_color_ext(dir_in, sun_dir, sun_color, vec3<f32>(0.09, 0.19, 0.42), vec3<f32>(0.62, 0.70, 0.82), 5.0);
}

fn sky_color_ext(dir_in: vec3<f32>, sun_dir: vec3<f32>, sun_color: vec3<f32>, zenith: vec3<f32>, horizon: vec3<f32>, falloff: f32) -> vec3<f32> {
    let dir = normalize(dir_in);
    let up = max(dir.y, 0.0);
    let t = pow(1.0 - up, max(falloff, 0.5));
    var sky = mix(zenith, horizon, t);
    let sun_amount = max(dot(dir, sun_dir), 0.0);
    sky += sun_color * pow(sun_amount, 8.0) * 0.4;
    sky += sun_color * smoothstep(0.9995, 0.9999, sun_amount) * 60.0;
    // blend into the ground half instead of switching at dir.y == 0, which otherwise leaves a
    // hard pale line across the sky exactly at the horizon
    let ground = mix(horizon * 0.80, vec3<f32>(0.045, 0.040, 0.035), clamp(-dir.y * 2.5, 0.0, 1.0));
    sky = mix(sky, ground, clamp(-dir.y * 14.0, 0.0, 1.0));
    return max(sky, vec3<f32>(0.0));
}

// ---- procedural surface detail ------------------------------------------------------------
// Value noise + a 3-octave sum, used to perturb shading normals from world position so ground,
// rock and water get micro-relief without a second UV set or another texture fetch.
fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn detail_fbm(p: vec2<f32>) -> f32 {
    return 0.52 * value_noise(p)
        + 0.30 * value_noise(p * 2.17 + vec2<f32>(3.1, 1.7))
        + 0.18 * value_noise(p * 4.43 + vec2<f32>(7.3, 5.1));
}

/// Perturb `n0` with the gradient of the detail field. `d` is (wavelength, strength, animate).
fn apply_detail(world_pos: vec3<f32>, view_depth: f32, n0: vec3<f32>, d: vec4<f32>, time: f32) -> vec3<f32> {
    if d.x <= 0.0 || d.y <= 0.0 {
        return n0;
    }
    var p = world_pos.xz / d.x;
    if d.z > 0.5 {
        p = p + vec2<f32>(time * 0.45, time * 0.26);
    }
    let e = 0.12;
    let h0 = detail_fbm(p);
    let gx = (detail_fbm(p + vec2<f32>(e, 0.0)) - h0) / e;
    let gz = (detail_fbm(p + vec2<f32>(0.0, e)) - h0) / e;
    let up = vec3<f32>(0.0, 1.0, 0.0);
    var t = cross(up, n0);
    if length(t) < 1e-3 {
        t = vec3<f32>(1.0, 0.0, 0.0);
    }
    t = normalize(t);
    let b = cross(n0, t);
    // fade out where a texel spans many ripples, otherwise it shimmers
    let fade = 1.0 - smoothstep(400.0, 5000.0, view_depth);
    var n1 = n0 - d.y * fade * 0.06 * (t * gx + b * gz);
    // a second, much finer octave that only shows close up (grass blades, gravel, ripples)
    let near_fade = 1.0 - smoothstep(20.0, 180.0, view_depth);
    if near_fade > 0.0 {
        var q = world_pos.xz / max(d.x * 0.09, 0.12);
        if d.z > 0.5 {
            q = q + vec2<f32>(time * 0.9, time * 0.5);
        }
        let g0 = detail_fbm(q);
        let qx = (detail_fbm(q + vec2<f32>(e, 0.0)) - g0) / e;
        let qz = (detail_fbm(q + vec2<f32>(0.0, e)) - g0) / e;
        n1 = n1 - d.y * near_fade * 0.035 * (t * qx + b * qz);
    }
    return normalize(n1);
}
