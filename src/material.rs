//! Materials: the PBR description, the five texture slots it feeds, and the bind group that binds
//! them.
//!
//! A material is a [`MaterialDesc`] (plain values, cheap to build and clone) plus a
//! [`MaterialMaps`] bundle of decoded images. [`Material::new`] uploads the maps — or neutral
//! 1×1 stand-ins for the ones left empty — so every material has exactly the same five texture
//! slots and the shader needs no permutation for "no normal map". The renderer binds a material
//! through [`create_material_bind_group`] once per frame at most, caching the result.

use anyhow::Result;

use crate::gpu;

/// A GPU texture with the view the material bind group needs.
///
/// Colour maps (albedo, emissive) are uploaded as `Rgba8UnormSrgb` so the hardware converts them
/// to linear on sample; data maps (normal, metallic-roughness, occlusion) stay `Rgba8Unorm`
/// because their channels are numbers, not colours. Mixing that up is the classic reason a normal
/// map looks "washed out" or a roughness map feels too shiny.
pub struct Texture {
    /// The texture itself; kept so mip levels and the view remain valid.
    #[allow(dead_code)]
    pub texture: wgpu::Texture,
    /// Full-chain view (every mip level) that the bind group binds.
    pub view: wgpu::TextureView,
}

impl Texture {
    /// Upload an RGBA8 image with a full mip pyramid generated on the CPU.
    ///
    /// Mips are built with [`gpu::generate_mips_rgba8`] and uploaded level by level, which is what
    /// makes minified textures (a distant runway, a wall seen at a glancing angle) resolve to a
    /// stable average instead of shimmering aliasing.
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

    /// Upload a single-pixel texture of one colour, used as the stand-in for a missing map.
    ///
    /// One pixel and one mip level is enough: every tap returns the same value, so the cost is a
    /// few bytes per material and one sampler fetch that the texture cache handles trivially.
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

    /// The neutral normal map: `(128, 128, 255)`, i.e. tangent-space "straight out".
    pub fn flat_normal(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_color(device, queue, "flat-normal", [128, 128, 255, 255], false)
    }

    /// Opaque white — the stand-in for missing albedo and occlusion maps.
    pub fn white(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_color(device, queue, "white", [255, 255, 255, 255], false)
    }

    /// Opaque black — the stand-in for a missing emissive map (adds nothing).
    pub fn black(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_color(device, queue, "black", [0, 0, 0, 255], false)
    }

    /// White metallic-roughness, i.e. a fully rough non-metal unless the shader scales it by the
    /// per-material `metallic`/`roughness` values.
    pub fn default_mr(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::from_color(device, queue, "default-mr", [255, 255, 255, 255], false)
    }
}

/// The scalar description of a surface — everything about a material that is not an image.
///
/// Values follow the glTF 2.0 metallic-roughness model, with the extra flags this renderer adds
/// for its own needs (dithering, procedural detail, forward transparency). Defaults are a matte
/// grey dielectric: safe to `..Default::default()` and override one field.
#[derive(Clone)]
pub struct MaterialDesc {
    /// Debug name; also the bind-group label.
    pub name: String,
    /// Linear base colour, RGBA. Alpha is coverage for `dithered`/`transparent` materials.
    pub base_color: [f32; 4],
    /// Linear emissive colour; multiplied by `emissive_strength` and the emissive map.
    pub emissive: [f32; 3],
    /// HDR multiplier on `emissive` — above 1 for lights, indicators and afterburners.
    pub emissive_strength: f32,
    /// Metalness, 0 (dielectric) to 1 (conductor). Metals take their tint from `base_color`.
    pub metallic: f32,
    /// Perceptual roughness, 0 (mirror) to 1 (fully diffuse).
    pub roughness: f32,
    /// Ambient-occlusion multiplier applied to the IBL diffuse term.
    pub ao: f32,
    /// Discard fragments whose alpha is below this value. 0 disables alpha testing.
    pub alpha_cutoff: f32,
    /// UV tiling factor applied on top of the mesh UVs.
    pub uv_scale: [f32; 2],
    /// UV offset applied after scaling (glTF `KHR_texture_transform`).
    pub uv_offset: [f32; 2],
    /// Render both faces, flipping the normal on back faces. Needed for planes, foliage cards and
    /// open hulls.
    pub double_sided: bool,
    /// Skip lighting entirely and show the base colour/texture as-is (glTF `KHR_materials_unlit`).
    pub unlit: bool,
    /// Treat base-colour/texture alpha as coverage via ordered dithering (fake transparency
    /// for the deferred pipeline: glass, smoke, cloud edges).
    pub dithered: bool,
    /// Wavelength in metres of the procedural micro-relief applied in the lighting pass
    /// (0 disables it). Adds surface grain without a second UV set.
    pub detail_scale: f32,
    /// Steepness of that micro-relief (0.3 subtle, 1.0 strong).
    pub detail_strength: f32,
    /// Scroll the detail pattern with time — water ripples.
    pub detail_anim: f32,
    /// Draw in the forward pass with real alpha blending (glass, clouds, smoke) instead of the
    /// deferred path, which can only dither.
    pub transparent: bool,
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
            detail_scale: 0.0,
            detail_strength: 0.0,
            detail_anim: 0.0,
            transparent: false,
        }
    }
}

/// Decoded images to upload into a [`Material`]'s texture slots.
///
/// All optional: a `None` slot becomes a neutral stand-in, so a procedural material needs no
/// images at all. Images are RGBA8 `image` buffers, which is what both the procedural generators
/// and [`crate::scene::loader`] produce.
#[derive(Default)]
pub struct MaterialMaps {
    /// Base colour / albedo, sampled as sRGB.
    pub albedo: Option<image::RgbaImage>,
    /// Tangent-space normal map, sampled linearly.
    pub normal: Option<image::RgbaImage>,
    /// glTF convention: green = roughness, blue = metallic, sampled linearly.
    pub metallic_roughness: Option<image::RgbaImage>,
    /// Emissive colour, sampled as sRGB.
    pub emissive: Option<image::RgbaImage>,
    /// Ambient occlusion in the red channel, sampled linearly.
    pub occlusion: Option<image::RgbaImage>,
}

/// An uploaded material: its description, its five textures and (once bound) its bind group.
///
/// The `has_*` flags record which maps were actually supplied, so shaders can skip a texture fetch
/// that would only return a constant — the normal map in particular costs a tangent-space
/// transform and a fetch per fragment.
pub struct Material {
    /// The scalar parameters this material was created with.
    pub desc: MaterialDesc,
    /// Base colour / albedo texture (white if none was supplied).
    pub albedo: Texture,
    /// Tangent-space normal map (flat if none was supplied).
    pub normal: Texture,
    /// Roughness in green, metalness in blue (white if none was supplied).
    pub metallic_roughness: Texture,
    /// Emissive colour (black if none was supplied).
    pub emissive_tex: Texture,
    /// Ambient occlusion in red (white if none was supplied).
    pub occlusion: Texture,
    /// Whether `albedo` came from a real image rather than the stand-in.
    pub has_albedo: bool,
    /// Whether `normal` came from a real image rather than the stand-in.
    pub has_normal: bool,
    /// Whether `metallic_roughness` came from a real image rather than the stand-in.
    pub has_mr: bool,
    /// Whether `emissive_tex` came from a real image rather than the stand-in.
    pub has_emissive: bool,
    /// Whether `occlusion` came from a real image rather than the stand-in.
    pub has_occlusion: bool,
    /// Bind group created by the renderer when the material is first used; `None` until then, and
    /// rebuilt whenever the material list is re-uploaded.
    pub bind_group: Option<wgpu::BindGroup>,
}

impl Material {
    /// Upload the supplied maps and fill the rest with neutral stand-ins.
    ///
    /// Returns an error only if a texture allocation or upload fails; a material with no maps at
    /// all succeeds, which is what the procedural scene relies on.
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

/// Create the sampler shared by every material texture.
///
/// Repeat addressing (so `uv_scale` can tile a small texture across a large surface), trilinear
/// minification, and `max_aniso`-tap anisotropic filtering when the adapter supports it — 16 on a
/// desktop GPU, which is what keeps a runway or a road from turning into a grey smear at grazing
/// angles. `max_aniso` is clamped to at least 1 so an adapter reporting no support still gets a
/// valid sampler.
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

/// Layout of the per-material bind group: five textures (bindings 0-4) and one sampler (binding 5).
///
/// The binding order is fixed and shared by every pipeline that shades geometry, which is why the
/// same bind group can be reused by the opaque, transparent and shadow passes.
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

/// Bind a material's textures and the shared sampler into one bind group.
///
/// The renderer caches the result in [`Material::bind_group`] and rebuilds it only when the scene's
/// material list changes, so this is not a per-frame allocation in steady state.
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
