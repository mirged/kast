use anyhow::Result;

use crate::gpu;

pub struct Texture {
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl Texture {
    pub fn from_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        image: &image::RgbaImage,
        srgb: bool,
    ) -> Result<Self> {
        let format = if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let levels = gpu::generate_mips_rgba8(image);
        let mip_count = levels.len() as u32;
        let texture = gpu::texture_2d(
            device,
            label,
            image.width(),
            image.height(),
            format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            mip_count,
            1,
        );
        for (level, (w, h, data)) in levels.iter().enumerate() {
            gpu::upload_2d(queue, &texture, 0, level as u32, *w, *h, data, w * 4);
        }
        let view = gpu::default_view(&texture);
        Ok(Self { texture, view })
    }

    pub fn from_color(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        color: [u8; 4],
        srgb: bool,
    ) -> Self {
        let format = if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };
        let texture = gpu::texture_2d(
            device,
            label,
            1,
            1,
            format,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            1,
            1,
        );
        gpu::upload_2d(queue, &texture, 0, 0, 1, 1, &color, 4);
        let view = gpu::default_view(&texture);
        Self { texture, view }
    }

    pub fn flat_normal(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_color(device, queue, "flat-normal", [128, 128, 255, 255], false)
    }

    pub fn white(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_color(device, queue, "white", [255, 255, 255, 255], false)
    }

    pub fn black(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_color(device, queue, "black", [0, 0, 0, 255], false)
    }

    pub fn default_mr(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_color(device, queue, "default-mr", [255, 255, 255, 255], false)
    }
}

#[derive(Clone)]
pub struct MaterialDesc {
    pub name: String,
    pub base_color: [f32; 4],
    pub emissive: [f32; 3],
    pub emissive_strength: f32,
    pub metallic: f32,
    pub roughness: f32,
    pub ao: f32,
    pub alpha_cutoff: f32,
    pub uv_scale: [f32; 2],
    pub uv_offset: [f32; 2],
    pub double_sided: bool,
    pub unlit: bool,
    /// Treat base-colour/texture alpha as coverage via ordered dithering (fake transparency
    /// for the deferred pipeline: glass, smoke, cloud edges).
    pub dithered: bool,
}

impl Default for MaterialDesc {
    fn default() -> Self {
        Self {
            name: "material".into(),
            base_color: [0.8, 0.8, 0.8, 1.0],
            emissive: [0.0, 0.0, 0.0],
            emissive_strength: 1.0,
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            alpha_cutoff: 0.0,
            uv_scale: [1.0, 1.0],
            uv_offset: [0.0, 0.0],
            double_sided: false,
            unlit: false,
            dithered: false,
        }
    }
}

#[derive(Default)]
pub struct MaterialMaps {
    pub albedo: Option<image::RgbaImage>,
    pub normal: Option<image::RgbaImage>,
    pub metallic_roughness: Option<image::RgbaImage>,
    pub emissive: Option<image::RgbaImage>,
    pub occlusion: Option<image::RgbaImage>,
}

pub struct Material {
    pub desc: MaterialDesc,
    pub albedo: Texture,
    pub normal: Texture,
    pub metallic_roughness: Texture,
    pub emissive_tex: Texture,
    pub occlusion: Texture,
    pub has_albedo: bool,
    pub has_normal: bool,
    pub has_mr: bool,
    pub has_emissive: bool,
    pub has_occlusion: bool,
    pub bind_group: Option<wgpu::BindGroup>,
}

impl Material {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        desc: MaterialDesc,
        maps: MaterialMaps,
    ) -> Result<Self> {
        let has_albedo = maps.albedo.is_some();
        let has_normal = maps.normal.is_some();
        let has_mr = maps.metallic_roughness.is_some();
        let has_emissive = maps.emissive.is_some();
        let has_occlusion = maps.occlusion.is_some();

        let albedo = match maps.albedo {
            Some(img) => Texture::from_rgba8(device, queue, "albedo", &img, true)?,
            None => Texture::white(device, queue),
        };
        let normal = match maps.normal {
            Some(img) => Texture::from_rgba8(device, queue, "normal", &img, false)?,
            None => Texture::flat_normal(device, queue),
        };
        let metallic_roughness = match maps.metallic_roughness {
            Some(img) => Texture::from_rgba8(device, queue, "metallic-roughness", &img, false)?,
            None => Texture::default_mr(device, queue),
        };
        let emissive_tex = match maps.emissive {
            Some(img) => Texture::from_rgba8(device, queue, "emissive", &img, true)?,
            None => Texture::black(device, queue),
        };
        let occlusion = match maps.occlusion {
            Some(img) => Texture::from_rgba8(device, queue, "occlusion", &img, false)?,
            None => Texture::white(device, queue),
        };

        Ok(Self {
            desc,
            albedo,
            normal,
            metallic_roughness,
            emissive_tex,
            occlusion,
            has_albedo,
            has_normal,
            has_mr,
            has_emissive,
            has_occlusion,
            bind_group: None,
        })
    }
}

pub fn create_sampler(device: &wgpu::Device, label: &str, max_aniso: u16) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        lod_min_clamp: 0.0,
        lod_max_clamp: 32.0,
        compare: None,
        anisotropy_clamp: max_aniso.max(1),
        border_color: None,
    })
}

pub fn create_material_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let tex = |binding: u32, sample_type: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let f = wgpu::TextureSampleType::Float { filterable: true };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material-bgl"),
        entries: &[
            tex(0, f),
            tex(1, f),
            tex(2, f),
            tex(3, f),
            tex(4, f),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub fn create_material_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    material: &Material,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("material-{}", material.desc.name)),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&material.albedo.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&material.normal.view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&material.metallic_roughness.view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&material.emissive_tex.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&material.occlusion.view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}
