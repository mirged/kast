use glam::{Mat4, Vec3};

use crate::gpu;
use crate::types::CASCADE_COUNT;

pub const SHADOW_SIZE: u32 = 2048;
pub const SHADOW_STRIDE: u64 = 256;

pub struct Shadow {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// Non-comparison sampler, used by the PCSS blocker search to read raw map depth.
    pub depth_sampler: wgpu::Sampler,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub matrix_buffer: wgpu::Buffer,
    pub size: u32,
}

impl Shadow {
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

    pub fn layer_view(&self, layer: u32) -> wgpu::TextureView {
        gpu::array_layer_view(&self.texture, layer)
    }

    pub fn update(
        &self,
        queue: &wgpu::Queue,
        view: Mat4,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
        sun_dir: Vec3,
    ) -> [f32; 4] {
        let (matrices, splits) =
            compute_cascades(view, fov_y, aspect, near, far, sun_dir, self.size);
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

#[allow(deprecated)]
pub fn compute_cascades(
    view: Mat4,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
    sun_dir: Vec3,
    size: u32,
) -> ([Mat4; CASCADE_COUNT], [f32; 4]) {
    let inv_view = view.inverse();
    let tan_half = (fov_y * 0.5).tan();
    let lambda = 0.6;

    let mut splits = [0.0f32; CASCADE_COUNT];
    for i in 0..CASCADE_COUNT {
        let p = (i + 1) as f32 / CASCADE_COUNT as f32;
        let log = near * (far / near).powf(p);
        let uni = near + (far - near) * p;
        splits[i] = lambda * log + (1.0 - lambda) * uni;
    }

    let light_dir = sun_dir.normalize();
    let up = if light_dir.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };

    let mut matrices = [Mat4::IDENTITY; CASCADE_COUNT];
    let mut prev = near;
    for i in 0..CASCADE_COUNT {
        let n = prev;
        let f = splits[i];
        prev = f;

        let mut corners = [Vec3::ZERO; 8];
        let mut idx = 0;
        for d in [n, f] {
            let hh = tan_half * d;
            let hw = hh * aspect;
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
