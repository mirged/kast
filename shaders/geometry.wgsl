@group(0) @binding(0) var<uniform> globals: Globals;
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
    @location(2) world_tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) view_depth: f32,
    @location(5) @interpolate(flat) obj_index: u32,
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
    out.world_tangent = vec4<f32>((obj.model * vec4<f32>(tangent.xyz, 0.0)).xyz, tangent.w);
    out.uv = uv * obj.uv_transform.xy + obj.uv_transform.zw;
    out.view_depth = -view.z;
    out.obj_index = obj_index;
    return out;
}

struct FsOut {
    @location(0) albedo_metal: vec4<f32>,
    @location(1) normal_rough: vec4<f32>,
    @location(2) emissive_ao: vec4<f32>,
};

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front_facing: bool) -> FsOut {
    let obj = objects[in.obj_index];

    let albedo_sample = textureSample(albedo_tex, tex_sampler, in.uv);
    let mr_sample = textureSample(mr_tex, tex_sampler, in.uv);
    let ao_sample = textureSample(occlusion_tex, tex_sampler, in.uv);
    let emissive_sample = textureSample(emissive_tex, tex_sampler, in.uv);
    let normal_sample = textureSample(normal_tex, tex_sampler, in.uv);

    var albedo = obj.base_color.rgb;
    if obj.flags0.x == 1u {
        albedo = albedo * albedo_sample.rgb;
    }

    var metallic = obj.material.x;
    var roughness = obj.material.y;
    if obj.flags0.z == 1u {
        roughness = roughness * mr_sample.g;
        metallic = metallic * mr_sample.b;
    }

    var ao = obj.material.z;
    if obj.flags1.x == 1u {
        ao = ao * ao_sample.r;
    }

    var emissive = obj.emissive.rgb * obj.emissive.a;
    if obj.flags0.w == 1u {
        emissive = emissive * emissive_sample.rgb;
    }

    var n = normalize(in.world_normal);
    if obj.flags0.y == 1u {
        let sampled = normal_sample.xyz * 2.0 - 1.0;
        let t = normalize(in.world_tangent.xyz);
        let b = normalize(cross(n, t)) * in.world_tangent.w;
        let tbn = mat3x3<f32>(t, b, n);
        n = normalize(tbn * sampled);
    }
    if front_facing == false && obj.flags1.z == 1u {
        n = -n;
    }

    let base_alpha = albedo_sample.a * obj.base_color.a;
    if obj.material.w > 0.0 && base_alpha < obj.material.w {
        discard;
    }
    // Ordered-dither (screen-door) transparency for the deferred path. The 4x4 Bayer pattern is
    // offset per object so overlapping translucent objects don't hide each other identically.
    if obj.flags1.w == 1u {
        let o = in.obj_index * 7u;
        let px = (u32(in.clip.x) + o) & 3u;
        let py = (u32(in.clip.y) + (o >> 2u)) & 3u;
        var bayer = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
        let threshold = (bayer[py * 4u + px] + 0.5) / 16.0;
        if base_alpha < threshold {
            discard;
        }
    }

    var out: FsOut;
    if obj.flags1.y == 1u {
        out.albedo_metal = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.normal_rough = vec4<f32>(n, 1.0);
        out.emissive_ao = vec4<f32>(albedo, 1.0);
    } else {
        out.albedo_metal = vec4<f32>(albedo, metallic);
        out.normal_rough = vec4<f32>(n, roughness);
        out.emissive_ao = vec4<f32>(emissive, ao);
    }
    return out;
}
