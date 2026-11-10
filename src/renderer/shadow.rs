//! Cascaded sun shadows: the shadow map, its samplers, and the frustum-fitting maths.
//!
//! One depth texture holds all [`CASCADE_COUNT`] cascades as array layers. Each cascade covers a
//! slice of the camera frustum, and the slices are chosen with a blend of logarithmic and uniform
//! distribution so near geometry gets fine shadow texels without the far slice collapsing into
//! aliasing at the horizon.
//!
//! The light-space matrix per cascade is fitted to the eight frustum corners of that slice, then
//! snapped to whole shadow texels: without that snap, camera motion makes every shadow edge crawl
//! by a fraction of a texel each frame.

use glam::{Mat4, Vec3};

use crate::gpu;
use crate::types::CASCADE_COUNT;

/// Edge length of each cascade's shadow map, in texels.
///
/// Bigger maps soften the compromise between reach and resolution but cost fill rate in the shadow
/// pass; 2048 is the point where a 4-cascade setup still looks crisp on a 1440p frame.
pub const SHADOW_SIZE: u32 = 2048;

/// Stride, in bytes, between cascade matrices in the uniform buffer.
///
/// 64 bytes is the size of one `mat4x4<f32>`; the buffer is a dynamic-offset uniform array, so the
/// per-cascade matrix is selected at bind time with an offset instead of a separate buffer per
/// cascade.
pub const SHADOW_STRIDE: u64 = 256;

/// The camera parameters a cascade set is fitted to.
///
/// Bundled into one struct because all four travel together everywhere and because
/// `compute_cascades` would otherwise take seven positional floats, which is exactly the kind of
/// signature where the arguments get swapped.
#[derive(Clone, Copy, Debug)]
pub struct FrustumFit {
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Frame aspect ratio (width / height).
    pub aspect: f32,
    /// Camera near plane in metres; the first cascade starts here.
    pub near: f32,
    /// Far edge to fit, in metres — the camera's far plane clamped by the shadow range.
    pub far: f32,
}

/// The shadow map and everything needed to render into it and sample it.
pub struct Shadow {
    /// `Depth32Float` array texture, one layer per cascade.
    pub texture: wgpu::Texture,
    /// Full array view, bound by the lighting pass.
    pub view: wgpu::TextureView,
    /// Comparison sampler for hardware PCF during the shading lookup.
    pub sampler: wgpu::Sampler,
    /// Non-comparison sampler, used by the PCSS blocker search to read raw map depth.
    pub depth_sampler: wgpu::Sampler,
    /// Layout of the dynamic-offset matrix buffer.
    pub layout: wgpu::BindGroupLayout,
    /// Bind group for the matrix buffer.
    pub bind_group: wgpu::BindGroup,
    /// Uniform buffer holding one light-space matrix per cascade.
    pub matrix_buffer: wgpu::Buffer,
    /// Shadow-map edge length in texels, needed by the cascade fitting and the shader uniforms.
    pub size: u32,
}

impl Shadow {
    /// Allocate the shadow map, both samplers and the matrix buffer.
    ///
    /// Both samplers are needed: a comparison sampler gives free 2x2 PCF for the final visibility
    /// test, but percentage-closer soft shadows first have to search for blockers, and that search
    /// needs the stored depth values rather than a comparison result.
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let texture = gpu::texture_2d_array(
            device,
            "shadow-map",
            size,
            size,
            CASCADE_COUNT as u32,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            1,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow-array-view"),
            format: None,
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: Some(wgpu::CompareFunction::LessEqual),
            anisotropy_clamp: 1,
            border_color: None,
        });

        // Depth textures cannot be sampled with a filtering sampler, and the blocker search
        // needs the stored depth rather than a comparison result.
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow-depth-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: 0.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });

        let matrix_buffer = gpu::buffer_empty(
            device,
            "shadow-matrices",
            SHADOW_STRIDE * CASCADE_COUNT as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow-bg"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &matrix_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(64),
                }),
            }],
        });

        Self {
            texture,
            view,
            sampler,
            depth_sampler,
            layout,
            bind_group,
            matrix_buffer,
            size,
        }
    }

    /// A 2D view of one cascade, for use as a depth attachment in the shadow pass.
    pub fn layer_view(&self, layer: u32) -> wgpu::TextureView {
        gpu::array_layer_view(&self.texture, layer)
    }

    /// Refit the cascades to the camera and upload their matrices; returns the split distances.
    ///
    /// The returned array is fed to `Globals::shadow_splits`, which is what the lighting shader
    /// uses to decide which cascade a fragment belongs to.
    pub fn update(
        &self,
        queue: &wgpu::Queue,
        view: Mat4,
        fit: FrustumFit,
        sun_dir: Vec3,
    ) -> [f32; 4] {
        let (matrices, splits) = compute_cascades(view, fit, sun_dir, self.size);
        for (i, m) in matrices.iter().enumerate() {
            queue.write_buffer(
                &self.matrix_buffer,
                i as u64 * SHADOW_STRIDE,
                bytemuck::cast_slice(&m.to_cols_array()),
            );
        }
        splits
    }
}

/// Fit one light-space orthographic projection per cascade to the slices of the camera frustum.
///
/// Returns the matrices plus the far distance of each slice. The steps, in order:
///
/// 1. choose slice boundaries with a logarithmic/uniform blend (`lambda` = 0.6) between `near` and
///    `far`, so the near slice follows the camera's own depth distribution;
/// 2. rebuild the eight frustum corners of each slice in world space and take their bounding
///    sphere — a sphere does not change shape as the camera turns, which is what stops the shadow
///    from swimming when you look around;
/// 3. place a light "eye" outside that sphere along the reversed sun direction and look at the
///    centre, with an `up` vector chosen to avoid degenerating when the sun is straight overhead;
/// 4. snap the orthographic bounds to whole texels and pad the depth range, which removes the
///    remaining sub-texel crawl and keeps casters just outside the slice from being clipped away.
#[allow(deprecated)]
pub fn compute_cascades(
    view: Mat4,
    fit: FrustumFit,
    sun_dir: Vec3,
    size: u32,
) -> ([Mat4; CASCADE_COUNT], [f32; 4]) {
    let inv_view = view.inverse();
    let tan_half = (fit.fov_y * 0.5).tan();
    let lambda = 0.6;

    let mut splits = [0.0f32; CASCADE_COUNT];
    for (i, split) in splits.iter_mut().enumerate() {
        let p = (i + 1) as f32 / CASCADE_COUNT as f32;
        let log = fit.near * (fit.far / fit.near).powf(p);
        let uni = fit.near + (fit.far - fit.near) * p;
        *split = lambda * log + (1.0 - lambda) * uni;
    }

    let light_dir = sun_dir.normalize();
    let up = if light_dir.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };

    let mut matrices = [Mat4::IDENTITY; CASCADE_COUNT];
    let mut prev = fit.near;
    for i in 0..CASCADE_COUNT {
        let n = prev;
        let f = splits[i];
        prev = f;

        let mut corners = [Vec3::ZERO; 8];
        let mut idx = 0;
        for d in [n, f] {
            let hh = tan_half * d;
            let hw = hh * fit.aspect;
            for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                corners[idx] = Vec3::new(x * hw, y * hh, -d);
                idx += 1;
            }
        }
        let world: Vec<Vec3> = corners
            .iter()
            .map(|c| inv_view.transform_point3(*c))
            .collect();
        let center = world.iter().copied().sum::<Vec3>() / 8.0;
        let radius = world
            .iter()
            .map(|p| p.distance(center))
            .fold(0.0f32, f32::max);
        // Stand the light back far enough that nothing between it and the slice is clipped; the
        // extra 10 m covers large casters that sit just outside the frustum.
        let eye = center - light_dir * (radius + 10.0);
        let light_view = Mat4::look_at_rh(eye, center, up);

        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for w in &world {
            let l = light_view.transform_point3(*w);
            min = min.min(l);
            max = max.max(l);
        }

        let texel = (max.x - min.x) / size as f32;
        let center_ls = light_view.transform_point3(center);
        let snap_x = (center_ls.x / texel).round() * texel - center_ls.x;
        let snap_y = (center_ls.y / texel).round() * texel - center_ls.y;
        min.x += snap_x;
        max.x += snap_x;
        min.y += snap_y;
        max.y += snap_y;

        // Depth padding proportional to the slice, so a tall caster just outside the frustum still
        // makes it into the map instead of being clipped at the near plane.
        let z_margin = (max.z - min.z) * 0.5 + 20.0;
        max.z += z_margin;
        min.z -= z_margin;

        let ortho = Mat4::orthographic_rh(
            min.x,
            max.x,
            min.y,
            max.y,
            (-max.z).max(0.01),
            (-min.z).max(0.02),
        );
        matrices[i] = ortho * light_view;
    }

    (matrices, splits)
}
