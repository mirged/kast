// Forward pass for really transparent geometry (glass, clouds, smoke, glows). The deferred path
// cannot blend, so these are drawn after lighting into the HDR target with alpha blending, depth
// tested against the deferred depth buffer and sorted back-to-front on the CPU.

struct LightingUniform {
    params: vec4<f32>,
    params2: vec4<f32>,
    fog: vec4<f32>,
    fog_sun: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(4) var irradiance_tex: texture_cube<f32>;
@group(0) @binding(5) var prefilter_tex: texture_cube<f32>;
@group(0) @binding(6) var brdf_lut: texture_2d<f32>;
@group(0) @binding(7) var env_sampler: sampler;
@group(0) @binding(8) var<uniform> light_u: LightingUniform;
@group(0) @binding(9) var env_tex: texture_cube<f32>;

@group(1) @binding(0) var<storage, read> objects: array<ObjectData>;

@group(2) @binding(0) var albedo_tex: texture_2d<f32>;
@group(2) @binding(1) var normal_tex: texture_2d<f32>;
@group(2) @binding(2) var mr_tex: texture_2d<f32>;
@group(2) @binding(3) var emissive_tex: texture_2d<f32>;
@group(2) @binding(4) var occlusion_tex: texture_2d<f32>;
@group(2) @binding(5) var tex_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_depth: f32,
    @location(4) @interpolate(flat) obj_index: u32,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) obj_index: u32,
) -> VsOut {
    let obj = objects[obj_index];
    let world = obj.model * vec4<f32>(position, 1.0);
    let view = globals.view * world;
    var out: VsOut;
    out.clip = globals.proj * view;
    out.world_pos = world.xyz;
    out.world_normal = (obj.normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    out.uv = uv * obj.uv_transform.xy + obj.uv_transform.zw;
    out.view_depth = -view.z;
    out.obj_index = obj_index;
    return out;
}

fn fogged(color: vec3<f32>, world_pos: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    if light_u.fog.z < 0.5 {
        return color;
    }
    let cam = globals.camera_pos.xyz;
    let dist = length(world_pos - cam);
    let k = max(light_u.fog.y, 1e-7);
    let kdh = k * (world_pos.y - cam.y);
    var integral = 1.0;
    if abs(kdh) > 1e-4 {
        integral = (1.0 - exp(-kdh)) / kdh;
    }
    let tau = light_u.fog.x * exp(-k * max(cam.y, 0.0)) * dist * integral;
    let amount = min(1.0 - exp(-max(tau, 0.0)), light_u.fog.w);
    let horizon_dir = normalize(vec3<f32>(view_dir.x, max(view_dir.y, 0.004), view_dir.z));
    var fog_color = textureSampleLevel(env_tex, env_sampler, horizon_dir, 0.0).rgb;
    let to_sun = normalize(-globals.sun_dir.xyz);
    let mie = pow(max(dot(view_dir, to_sun), 0.0), max(light_u.fog_sun.w, 1.0));
    fog_color += light_u.fog_sun.rgb * mie;
    return mix(color, fog_color, amount);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    let obj = objects[in.obj_index];
    let albedo_sample = textureSample(albedo_tex, tex_sampler, in.uv);
    var albedo = obj.base_color.rgb;
    if obj.flags0.x == 1u {
        albedo = albedo * albedo_sample.rgb;
    }
    var alpha = obj.base_color.a;
    if obj.flags0.x == 1u {
        alpha = alpha * albedo_sample.a;
    }

    var n = normalize(in.world_normal);
    if obj.flags0.y == 1u {
        let sampled = textureSample(normal_tex, tex_sampler, in.uv).xyz * 2.0 - 1.0;
        n = normalize(n + sampled * 0.5);
    }
    if front_facing == false && obj.flags1.z == 1u {
        n = -n;
    }

    let v = normalize(globals.camera_pos.xyz - in.world_pos);
    let n_dot_v = max(dot(n, v), 1e-4);
    let metallic = obj.material.x;
    let roughness = clamp(obj.material.y, 0.03, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lit = vec3<f32>(0.0);
    let light_count = globals.light_meta.x;
    for (var i = 0u; i < light_count; i = i + 1u) {
        let light = lights[i];
        if light.params.w < 0.5 {
            continue;
        }
        var l: vec3<f32>;
        var radiance = light.color.rgb;
        if light.position.w < 0.5 {
            l = normalize(-light.direction.xyz);
        } else {
            let to_light = light.position.xyz - in.world_pos;
            let dist = length(to_light);
            if dist < 1e-4 {
                continue;
            }
            l = to_light / dist;
            let d2 = max(dist * dist, 1e-4);
            let window = clamp(1.0 - pow(dist / max(light.direction.w, 1e-3), 4.0), 0.0, 1.0);
            radiance = radiance * (window * window) / d2;
        }
        let n_dot_l = dot(n, l);
        if n_dot_l <= 0.0 {
            continue;
        }
        let h = normalize(v + l);
        let d = distribution_ggx(max(dot(n, h), 0.0), roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let f = fresnel_schlick(max(dot(v, h), 0.0), f0);
        let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
        let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
        lit += (kd * albedo * INV_PI + specular) * radiance * n_dot_l;
    }

    var ambient = vec3<f32>(0.0);
    if light_u.params.z > 0.5 {
        let r = reflect(-v, n);
        let f = fresnel_schlick_roughness(n_dot_v, f0, roughness);
        let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
        let irradiance = textureSampleLevel(irradiance_tex, env_sampler, n, 0.0).rgb;
        let prefiltered =
            textureSampleLevel(prefilter_tex, env_sampler, r, roughness * light_u.params2.y).rgb;
        let brdf = textureSampleLevel(brdf_lut, tex_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
        ambient = (kd * irradiance * albedo + prefiltered * (f * brdf.x + brdf.y)) * light_u.params.w;
    }
    ambient += globals.ambient.rgb * albedo;

    var emissive = obj.emissive.rgb * obj.emissive.a;
    if obj.flags0.w == 1u {
        emissive = emissive * textureSample(emissive_tex, tex_sampler, in.uv).rgb;
    }

    let view_dir = normalize(in.world_pos - globals.camera_pos.xyz);
    let color = fogged(lit + ambient + emissive, in.world_pos, view_dir);
    return vec4<f32>(color, clamp(alpha, 0.0, 1.0));
}
