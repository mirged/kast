struct LightingUniform {
    params: vec4<f32>,
    params2: vec4<f32>,
    fog: vec4<f32>,
    fog_sun: vec4<f32>,
};

// Exponential height fog with analytic optical depth plus a Mie-like sun lobe.
fn apply_fog(color: vec3<f32>, world_pos: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    if light_u.fog.z < 0.5 {
        return color;
    }
    let cam = globals.camera_pos.xyz;
    let dist = length(world_pos - cam);
    let k = max(light_u.fog.y, 1e-7);
    let dh = world_pos.y - cam.y;
    let kdh = k * dh;
    var integral = 1.0;
    if abs(kdh) > 1e-4 {
        integral = (1.0 - exp(-kdh)) / kdh;
    }
    let tau = light_u.fog.x * exp(-k * max(cam.y, 0.0)) * dist * integral;
    let amount = min(1.0 - exp(-max(tau, 0.0)), light_u.fog.w);
    // fog takes the colour of the sky just above the horizon in the same azimuth, so fogged
    // geometry meets the sky seamlessly; fog_sun.rgb is an optional extra sun-glow tint.
    let horizon_dir = normalize(vec3<f32>(view_dir.x, max(view_dir.y, 0.004), view_dir.z));
    var fog_color = textureSampleLevel(env_tex, env_sampler, horizon_dir, 0.0).rgb;
    let to_sun = normalize(-globals.sun_dir.xyz);
    let mie = pow(max(dot(view_dir, to_sun), 0.0), max(light_u.fog_sun.w, 1.0));
    fog_color += light_u.fog_sun.rgb * mie;
    return mix(color, fog_color, amount);
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(2) var shadow_map: texture_depth_2d_array;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;
@group(0) @binding(4) var irradiance_tex: texture_cube<f32>;
@group(0) @binding(5) var prefilter_tex: texture_cube<f32>;
@group(0) @binding(6) var brdf_lut: texture_2d<f32>;
@group(0) @binding(7) var env_sampler: sampler;
@group(0) @binding(8) var<uniform> light_u: LightingUniform;
@group(0) @binding(9) var env_tex: texture_cube<f32>;

@group(1) @binding(0) var g_albedo: texture_2d<f32>;
@group(1) @binding(1) var g_normal: texture_2d<f32>;
@group(1) @binding(2) var g_emissive: texture_2d<f32>;
@group(1) @binding(3) var g_depth: texture_depth_2d;
@group(1) @binding(4) var g_sampler: sampler;
@group(1) @binding(5) var ssao_tex: texture_2d<f32>;

fn point_attenuation(dist: f32, range: f32) -> f32 {
    let d2 = max(dist * dist, 1e-4);
    let window = clamp(1.0 - pow(dist / max(range, 1e-3), 4.0), 0.0, 1.0);
    return (window * window) / d2;
}

fn cascade_index(view_depth: f32) -> i32 {
    if view_depth < globals.shadow_splits.x {
        return 0;
    } else if view_depth < globals.shadow_splits.y {
        return 1;
    } else if view_depth < globals.shadow_splits.z {
        return 2;
    }
    return 3;
}

fn shadow_factor(world_pos: vec3<f32>, view_depth: f32, ndotl: f32) -> f32 {
    let cascade = cascade_index(view_depth);
    let lp = globals.cascade_matrices[u32(cascade)] * vec4<f32>(world_pos, 1.0);
    if lp.w <= 0.0 {
        return 1.0;
    }
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let texel = globals.shadow_texel.x;
    let bias = max(0.0012 * (1.0 - ndotl), 0.00025);
    let reference = ndc.z - bias;
    let layer = u32(cascade);
    var sum = 0.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel;
            sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, layer, reference);
        }
    }
    return sum / 9.0;
}

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4<f32> {
    let depth = textureSampleLevel(g_depth, g_sampler, in.uv, 0.0);
    let world_pos = reconstruct_world_pos(in.uv, depth, globals.inv_view_proj);
    // Reverse-Z with an infinite far plane puts the sky (depth 0) at w = 0, so derive the view
    // direction from a point at a tiny positive depth instead (~50 km away for near = 5 cm).
    let dir_pos = reconstruct_world_pos(in.uv, max(depth, 1e-6), globals.inv_view_proj);
    let view_dir = normalize(dir_pos - globals.camera_pos.xyz);

    if depth <= 0.0 {
        // sky: the baked environment cube, plus an analytic sun disc and its glow so the sun is
        // crisp at any resolution (the cube face is only 256 px across).
        var sky = textureSampleLevel(env_tex, env_sampler, view_dir, 0.0).rgb;
        let to_sun = normalize(-globals.sun_dir.xyz);
        let d = max(dot(view_dir, to_sun), 0.0);
        let disc = smoothstep(0.99985, 0.99995, d);
        let glow = pow(d, 900.0) * 1.2 + pow(d, 24.0) * 0.10;
        sky += globals.sun_color.rgb * (disc * 24.0 + glow);
        return vec4<f32>(sky, 1.0);
    }

    let albedo_metal = textureSampleLevel(g_albedo, g_sampler, in.uv, 0.0);
    let normal_rough = textureSampleLevel(g_normal, g_sampler, in.uv, 0.0);
    let emissive_ao = textureSampleLevel(g_emissive, g_sampler, in.uv, 0.0);

    let albedo = albedo_metal.rgb;
    let metallic = albedo_metal.a;
    let normal = normalize(normal_rough.xyz);
    let roughness = clamp(normal_rough.w, 0.04, 1.0);
    let emissive = emissive_ao.rgb;
    var ao = emissive_ao.a;
    if light_u.params.x > 0.5 {
        ao = ao * textureSampleLevel(ssao_tex, g_sampler, in.uv, 0.0).r;
    }

    let v = -view_dir;
    let n_dot_v = max(dot(normal, v), 1e-4);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    var lo = vec3<f32>(0.0);
    let light_count = globals.light_meta.x;
    for (var i = 0u; i < light_count; i = i + 1u) {
        let light = lights[i];
        if light.params.w < 0.5 {
            continue;
        }
        let kind = light.position.w;
        var l: vec3<f32>;
        var radiance = light.color.rgb;
        var atten = 1.0;

        if kind < 0.5 {
            l = normalize(-light.direction.xyz);
        } else {
            let to_light = light.position.xyz - world_pos;
            let dist = length(to_light);
            if dist < 1e-4 {
                continue;
            }
            l = to_light / dist;
            atten = point_attenuation(dist, light.direction.w);
            if kind > 1.5 {
                let cd = dot(normalize(light.direction.xyz), -l);
                let spot = clamp((cd - light.params.y) / max(light.params.x - light.params.y, 1e-4), 0.0, 1.0);
                atten = atten * spot * spot;
            }
        }

        let n_dot_l = dot(normal, l);
        if n_dot_l <= 0.0 {
            continue;
        }

        var shadow = 1.0;
        if kind < 0.5 && light_u.params.y > 0.5 {
            let world_view_depth = -(globals.view * vec4<f32>(world_pos, 1.0)).z;
            shadow = shadow_factor(world_pos, world_view_depth, n_dot_l);
            shadow = mix(1.0, shadow, light_u.params2.x);
        }

        let h = normalize(v + l);
        let n_dot_h = max(dot(normal, h), 0.0);
        let v_dot_h = max(dot(v, h), 0.0);

        let d = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let f = fresnel_schlick(v_dot_h, f0);
        let specular = (d * g * f) / max(4.0 * n_dot_v * n_dot_l, 1e-4);
        let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);

        lo += (kd * albedo * INV_PI + specular) * radiance * n_dot_l * atten * shadow;
    }

    var ambient = vec3<f32>(0.0);
    if light_u.params.z > 0.5 {
        let r = reflect(-v, normal);
        let f = fresnel_schlick_roughness(n_dot_v, f0, roughness);
        let kd = (vec3<f32>(1.0) - f) * (1.0 - metallic);
        let irradiance = textureSampleLevel(irradiance_tex, env_sampler, normal, 0.0).rgb;
        let diffuse = irradiance * albedo;
        let max_mip = light_u.params2.y;
        let prefiltered = textureSampleLevel(prefilter_tex, env_sampler, r, roughness * max_mip).rgb;
        let brdf = textureSampleLevel(brdf_lut, g_sampler, vec2<f32>(n_dot_v, roughness), 0.0).rg;
        let specular = prefiltered * (f * brdf.x + brdf.y);
        ambient = (kd * diffuse + specular) * light_u.params.w;
    }

    ambient = ambient * ao;
    ambient += globals.ambient.rgb * albedo * ao;

    let color = lo + ambient + emissive;
    return vec4<f32>(apply_fog(color, world_pos, view_dir), 1.0);
}
