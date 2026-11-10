use bytemuck::{Pod, Zeroable};

pub const MAX_LIGHTS: usize = 64;
pub const CASCADE_COUNT: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Globals {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub screen: [f32; 4],
    pub near_far: [f32; 4],
    pub ambient: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub light_meta: [u32; 4],
    pub time: [f32; 4],
    pub shadow_splits: [f32; 4],
    /// x = 1 / shadow-map size, y = shadow-map size, z = PCSS softness, w unused
    pub shadow_texel: [f32; 4],
    pub cascade_matrices: [[[f32; 4]; 4]; CASCADE_COUNT],
}

impl Default for Globals {
    fn default() -> Self {
        Self {
            view: glam::Mat4::IDENTITY.to_cols_array_2d(),
            proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            inv_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 4],
            screen: [1.0, 1.0, 1.0, 1.0],
            near_far: [0.1, 1000.0, 0.0, 0.0],
            ambient: [0.03, 0.03, 0.04, 1.0],
            sun_dir: [0.0, 1.0, 0.0, 0.0],
            sun_color: [1.0, 1.0, 1.0, 1.0],
            light_meta: [0; 4],
            time: [0.0; 4],
            shadow_splits: [10.0, 25.0, 60.0, 150.0],
            shadow_texel: [1.0 / 2048.0, 2048.0, 0.0, 0.0],
            cascade_matrices: [[[0.0; 4]; 4]; CASCADE_COUNT],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuLight {
    /// xyz = position, w = type (0 = directional, 1 = point, 2 = spot)
    pub position: [f32; 4],
    /// xyz = direction, w = range
    pub direction: [f32; 4],
    /// rgb = color * intensity, a = radius
    pub color: [f32; 4],
    /// x = inner cone cos, y = outer cone cos, z = shadow index, w = enabled
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ObjectData {
    pub model: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
    pub base_color: [f32; 4],
    /// rgb = emissive color, a = emissive strength
    pub emissive: [f32; 4],
    /// x = metallic, y = roughness, z = ao, w = alpha cutoff
    pub material: [f32; 4],
    /// xy = uv scale, zw = uv offset
    pub uv_transform: [f32; 4],
    /// x = has albedo, y = has normal, z = has mr, w = has emissive
    pub flags0: [u32; 4],
    /// x = has ao, y = unlit, z = double_sided, w = dithered transparency
    pub flags1: [u32; 4],
    /// x = detail wavelength (metres, 0 = off), y = strength, z = animate with time
    pub detail: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SkyUniform {
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    /// x = turbidity, y = exposure, z = face index, w = mode
    pub params: [f32; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    /// rgb = zenith colour, a = horizon falloff exponent
    pub zenith: [f32; 4],
    /// rgb = horizon colour
    pub horizon: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct LightingUniform {
    /// x = ssao enabled, y = shadows enabled, z = ibl enabled, w = ibl intensity
    pub params: [f32; 4],
    /// x = shadow strength, y = max mip, z = debug mode, w = ambient intensity
    pub params2: [f32; 4],
    /// x = fog density (1/m at sea level), y = height falloff (1/m), z = enabled, w = max opacity
    pub fog: [f32; 4],
    /// rgb = sun in-scatter tint, a = mie anisotropy exponent
    pub fog_sun: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct IblUniform {
    /// x = roughness, y = mip level, z = sample count, w = face size
    pub params: [f32; 4],
    /// x = face index, y = mip count, z/w unused
    pub face: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SsaoUniform {
    pub proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    /// x = radius, y = bias, z = intensity, w = power
    pub params: [f32; 4],
    /// x = sample count, y = width, z = height, w = time
    pub screen: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SsrUniform {
    pub proj: [[f32; 4]; 4],
    pub inv_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    /// x = max steps, y = thickness, z = stride, w = intensity
    pub params: [f32; 4],
    /// x = width, y = height, z = frame, w = max_distance
    pub screen: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct PostUniform {
    /// x = exposure, y = gamma, z = bloom intensity, w = vignette
    pub params: [f32; 4],
    /// x = tonemap mode, y = enable bloom, z = enable ssao, w = enable ssr
    pub modes: [f32; 4],
    /// x = width, y = height, z = 1/width, w = 1/height
    pub screen: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BloomUniform {
    /// x = source mip texel width, y = source mip texel height
    /// z = threshold, w = soft knee
    pub params: [f32; 4],
    /// x = target width, y = target height, z = level, w = is_upsample
    pub screen: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct DofUniform {
    pub inv_proj: [[f32; 4]; 4],
    /// x = focus distance, y = focal range, z = max radius, w = aperture
    pub params: [f32; 4],
    /// x = near, y = far, z = width, w = height
    pub depth: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct FxaaUniform {
    /// x = 1/width, y = 1/height, z = enabled, w = gamma encode
    pub params: [f32; 4],
}
