//! The GPU-side uniform and storage layout shared with the WGSL shaders.
//!
//! Everything here is `#[repr(C)]` + [`bytemuck::Pod`] and is written straight into a buffer, so
//! field order, field types and padding must match the corresponding `struct` in
//! `shaders/common.wgsl` (and the per-pass headers) exactly. A field added here without the
//! matching WGSL change does not fail to compile — it silently shifts every following value — so
//! these structs are the single place to change when a shader input changes.
//!
//! The trailing `[f32; 4]` / `[u32; 4]` fields are not padding: each one is a small vector of
//! scalars packed by meaning (documented per field), which keeps the buffer compact and the
//! binding count low while still being std140-friendly (every member is 4- or 16-byte aligned).

use bytemuck::{Pod, Zeroable};

/// Maximum number of punctual lights uploaded per frame.
///
/// The light buffer is allocated once for this count and the shader loops over `light_meta.x`
/// entries, so raising this costs memory but no pipeline change — while raising it past what a
/// backend's storage buffer allows is a hard limit, not a compile error.
pub const MAX_LIGHTS: usize = 64;

/// Number of sun-shadow cascades. Fixed because the cascade splits and matrices are packed into
/// 4-wide vectors, and because four cascades is the point where the extra maps stop paying for
/// themselves at this shadow-map size.
pub const CASCADE_COUNT: usize = 4;

/// Per-frame globals bound to every pass that needs the camera: geometry, lighting, SSR, DoF.
///
/// Matrices are column-major `[[f32; 4]; 4]`, which is what WGSL's `mat4x4<f32>` expects and what
/// `glam` produces, so no transposition happens anywhere in the pipeline.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct Globals {
    /// World-to-view matrix.
    pub view: [[f32; 4]; 4],
    /// Reverse-Z infinite projection (depth 1 at `z_near`, 0 at infinity).
    pub proj: [[f32; 4]; 4],
    /// `proj * view`, precomputed for the vertex shaders.
    pub view_proj: [[f32; 4]; 4],
    /// Inverse of `view_proj`, for reconstructing world positions from depth.
    pub inv_view_proj: [[f32; 4]; 4],
    /// `xyz` = camera position in world space, `w` unused.
    pub camera_pos: [f32; 4],
    /// `xy` = render width/height in pixels, `zw` = their reciprocals.
    pub screen: [f32; 4],
    /// `x` = near plane, `y` = far plane (used for culling and fog, not for the projection),
    /// `zw` unused.
    pub near_far: [f32; 4],
    /// `xyz` = ambient light colour, `w` = ambient intensity.
    pub ambient: [f32; 4],
    /// `xyz` = unit vector toward the sun, `w` unused.
    pub sun_dir: [f32; 4],
    /// `xyz` = sun colour × intensity, `w` unused.
    pub sun_color: [f32; 4],
    /// `x` = active light count, `y` = whether any shadow cascade is valid, `zw` unused.
    pub light_meta: [u32; 4],
    /// `x` = accumulated seconds since the renderer was created (drives animated detail and
    /// dither patterns), `yzw` unused.
    pub time: [f32; 4],
    /// Far edge of each cascade in view space, in metres (`x` = cascade 0 … `w` = cascade 3).
    pub shadow_splits: [f32; 4],
    /// x = 1 / shadow-map size, y = shadow-map size, z = PCSS softness, w unused
    pub shadow_texel: [f32; 4],
    /// World-to-light matrix per cascade, indexed 0..[`CASCADE_COUNT`].
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

/// One punctual light as the shader sees it: type-tagged, packed into four vectors.
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

/// One drawable object, bound as a storage buffer and indexed by the per-instance slot id.
///
/// This is the deferred pipeline's replacement for per-instance matrices: the geometry pass writes
/// only a `u32` per instance, and every per-object datum lives here. Fields are grouped so that
/// the common ones (`model`, `normal_matrix`) come first and each 16-byte vector is fully used.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct ObjectData {
    /// Object-to-world matrix.
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose of the model's upper 3×3, for transforming normals under non-uniform
    /// scale. Stored as a full 4×4 to keep the std140 layout simple.
    pub normal_matrix: [[f32; 4]; 4],
    /// Linear base colour, alpha included.
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

/// Inputs for the sky pass, which renders the environment cube face by face and then bakes the IBL
/// cubemaps from the same shader with a different `mode`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SkyUniform {
    /// xyz = unit vector toward the sun.
    pub sun_direction: [f32; 4],
    /// rgb = sun colour × intensity.
    pub sun_color: [f32; 4],
    /// x = turbidity, y = exposure, z = face index, w = mode
    pub params: [f32; 4],
    /// Inverse of the (cube-face) view-projection, for reconstructing the view ray.
    pub inv_view_proj: [[f32; 4]; 4],
    /// xyz = camera position, so the sky follows the eye when used as a background.
    pub camera_pos: [f32; 4],
    /// rgb = zenith colour, a = horizon falloff exponent
    pub zenith: [f32; 4],
    /// rgb = horizon colour
    pub horizon: [f32; 4],
}

/// Inputs for the deferred lighting pass: the feature toggles and the fog parameters.
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

/// Inputs for the IBL passes (irradiance convolution and radiance prefilter).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct IblUniform {
    /// x = roughness, y = mip level, z = sample count, w = face size
    pub params: [f32; 4],
    /// x = face index, y = mip count, z/w unused
    pub face: [u32; 4],
}

/// Inputs for the SSAO pass and its separable blur.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SsaoUniform {
    /// Camera projection (reverse-Z, infinite far).
    pub proj: [[f32; 4]; 4],
    /// Inverse projection, for turning depth back into view space.
    pub inv_proj: [[f32; 4]; 4],
    /// World-to-view.
    pub view: [[f32; 4]; 4],
    /// View-to-world.
    pub inv_view: [[f32; 4]; 4],
    /// x = radius, y = bias, z = intensity, w = power
    pub params: [f32; 4],
    /// x = sample count, y = width, z = height, w = time
    pub screen: [f32; 4],
}

/// Inputs for the screen-space reflection ray march.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct SsrUniform {
    /// Camera projection (reverse-Z, infinite far).
    pub proj: [[f32; 4]; 4],
    /// Inverse projection.
    pub inv_proj: [[f32; 4]; 4],
    /// World-to-view.
    pub view: [[f32; 4]; 4],
    /// x = max steps, y = thickness, z = stride, w = intensity
    pub params: [f32; 4],
    /// x = width, y = height, z = frame, w = max_distance
    pub screen: [f32; 4],
}

/// Inputs for the tonemap pass, which is also where exposure, bloom mix and gamma encoding happen.
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

/// Inputs for one bloom step; the same struct is reused for prefilter, downsample and upsample.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct BloomUniform {
    /// x = source mip texel width, y = source mip texel height
    /// z = threshold, w = soft knee
    pub params: [f32; 4],
    /// x = target width, y = target height, z = level, w = is_upsample
    pub screen: [f32; 4],
}

/// Inputs for the depth-of-field pass (a single-pass gather blur driven by CoC radius).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct DofUniform {
    /// Inverse projection, for linearising depth.
    pub inv_proj: [[f32; 4]; 4],
    /// x = focus distance, y = focal range, z = max radius, w = aperture
    pub params: [f32; 4],
    /// x = near, y = far, z = width, w = height
    pub depth: [f32; 4],
}

/// Inputs for the FXAA pass; also carries the flag that decides whether the final write still has
/// to encode gamma for a non-sRGB target.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct FxaaUniform {
    /// x = 1/width, y = 1/height, z = enabled, w = gamma encode
    pub params: [f32; 4],
}
