//! Every render target the frame needs, allocated in one place.
//!
//! Targets are sized in pixels and rebuilt wholesale on resize: they are cheap to recreate
//! compared with the pipelines that read them, and rebuilding avoids a class of stale-bind-group
//! bugs. Formats are chosen per target for precision where it matters — 8-bit for albedo, 16-bit
//! float for anything that carries linear HDR radiance, 32-bit float for reversed-Z depth.

use crate::gpu;

/// A render target: a texture, its default view, and the size it was allocated at.
///
/// The size is stored rather than queried from the texture because the bloom chain is allocated at
/// fractions of the screen size and every consumer needs to know its own extents (to compute texel
/// sizes for the blur uniforms).
pub struct Rt {
    /// The texture itself; `COPY_SRC` is only set on the present target.
    pub texture: wgpu::Texture,
    /// Default view over every mip level and layer.
    pub view: wgpu::TextureView,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Rt {
    /// Allocate a 2D render target with the given format, usage and mip count.
    ///
    /// `mip_levels` is 1 everywhere today; it exists because the bloom upsample chain and the
    /// reflection mips are the same struct and a mipmapped target is otherwise identical.
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
        mip_levels: u32,
    ) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let texture = gpu::texture_2d(device, label, width, height, format, usage, mip_levels, 1);
        let view = gpu::default_view(&texture);
        Self {
            texture,
            view,
            width,
            height,
        }
    }
}

/// Number of bloom levels: the first target is half resolution and each subsequent one halves
/// again, so six levels reach 1/64 of the screen — wide enough for a soft halo, cheap enough to
/// run every frame.
pub const BLOOM_LEVELS: usize = 6;

/// The full set of per-size targets owned by the renderer.
///
/// Lifetimes follow the passes: `g_*` and `depth` hold the G-buffer, `ssao_*` the occlusion term,
/// `hdr_a`/`hdr_b` ping-pong the lit image (DoF reads one and writes the other), `bloom` holds the
/// filter chain and `ldr` the tone-mapped image that FXAA reads last.
pub struct Targets {
    /// Width and height every full-resolution member was allocated at.
    pub size: (u32, u32),
    /// G-buffer: base colour, 8-bit (the deferred pass cannot use more precision usefully).
    pub g_albedo: Rt,
    /// G-buffer: world normal and roughness/metallic in 16-bit float.
    pub g_normal: Rt,
    /// G-buffer: emissive colour in 16-bit float, so emissive strength above 1 survives.
    pub g_emissive: Rt,
    /// Reversed-Z depth, `Depth32Float`, also sampled by SSAO, SSR and DoF.
    pub depth: Rt,
    /// Linear HDR lit image (scene + reflections + transparency).
    pub hdr_a: Rt,
    /// Second HDR target, so DoF can read `hdr_a` and write here.
    pub hdr_b: Rt,
    /// Tone-mapped, display-encoded image, the input to FXAA.
    pub ldr: Rt,
    /// Raw (noisy) SSAO, 8-bit single channel.
    pub ssao_raw: Rt,
    /// Blurred SSAO, what the lighting pass samples.
    pub ssao_blur: Rt,
    /// Bloom chain, index 0 is half resolution and each following level halves again.
    pub bloom: Vec<Rt>,
}

impl Targets {
    /// Allocate the whole set for a `width` × `height` drawable.
    ///
    /// Every target starts at `RENDER_ATTACHMENT | TEXTURE_BINDING`: all of them are both written
    /// and read within a frame. The bloom chain starts at half resolution because the first
    /// downsample does the same job more cheaply.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let color = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;

        let g_albedo = Rt::new(
            device,
            "g-albedo",
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            color,
            1,
        );
        let g_normal = Rt::new(
            device,
            "g-normal",
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            color,
            1,
        );
        let g_emissive = Rt::new(
            device,
            "g-emissive",
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            color,
            1,
        );
        let depth = Rt::new(
            device,
            "depth",
            width,
            height,
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            1,
        );
        let hdr_a = Rt::new(
            device,
            "hdr-a",
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            color,
            1,
        );
        let hdr_b = Rt::new(
            device,
            "hdr-b",
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            color,
            1,
        );
        let ldr = Rt::new(
            device,
            "ldr",
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            color,
            1,
        );
        let ssao_raw = Rt::new(
            device,
            "ssao-raw",
            width,
            height,
            wgpu::TextureFormat::R8Unorm,
            color,
            1,
        );
        let ssao_blur = Rt::new(
            device,
            "ssao-blur",
            width,
            height,
            wgpu::TextureFormat::R8Unorm,
            color,
            1,
        );

        let mut bloom = Vec::new();
        let mut bw = width / 2;
        let mut bh = height / 2;
        for i in 0..BLOOM_LEVELS {
            bloom.push(Rt::new(
                device,
                &format!("bloom-{i}"),
                bw,
                bh,
                wgpu::TextureFormat::Rgba16Float,
                color,
                1,
            ));
            bw = (bw / 2).max(1);
            bh = (bh / 2).max(1);
        }

        Self {
            size: (width, height),
            g_albedo,
            g_normal,
            g_emissive,
            depth,
            hdr_a,
            hdr_b,
            ldr,
            ssao_raw,
            ssao_blur,
            bloom,
        }
    }
}
