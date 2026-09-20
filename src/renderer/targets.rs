use crate::gpu;

pub struct Rt {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl Rt {
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

pub const BLOOM_LEVELS: usize = 6;

pub struct Targets {
    pub size: (u32, u32),
    pub g_albedo: Rt,
    pub g_normal: Rt,
    pub g_emissive: Rt,
    pub depth: Rt,
    pub hdr_a: Rt,
    pub hdr_b: Rt,
    pub ldr: Rt,
    pub ssao_raw: Rt,
    pub ssao_blur: Rt,
    pub bloom: Vec<Rt>,
}

impl Targets {
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
